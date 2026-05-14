# SPDX-License-Identifier: MIT
"""Visualize a single introspection record OR a CUDA-vs-CPU drift comparison.

Per CLAUDE.md "no /tmp paths in persisted artifacts": the caller passes an
explicit ``--output-dir``. Per "MPS auth eval is NOISE": MPS records render
with an explicit ``[advisory only]`` watermark.

Usage::

    .venv/bin/python tools/visualize_scorer_drift.py \\
        --record-a experiments/results/scorer_introspection_demo_.../posenet_record.pt \\
        --record-b experiments/results/scorer_introspection_demo_.../posenet_record_cuda.pt \\
        --output-dir reports/scorer_drift/<timestamp>/

If only ``--record-a`` is given, emits an architectural fingerprint figure
(layer-type histogram, mixer-rank-vs-depth, activation-numel distribution) for
that single record. CUDA-vs-CPU comparison emits the drift bar chart and the
compounding-factor plot in addition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.diagnostics import (  # noqa: E402
    IntrospectionRecord,
    compute_layer_drift,
)
from tac.diagnostics.cuda_cpu_drift import (  # noqa: E402
    drift_to_dict,
    estimate_compounding_for_path,
)


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"matplotlib not available ({exc}); install with `uv pip install matplotlib`"
        )


def _single_record_fingerprint_figures(record: IntrospectionRecord, out_dir: Path) -> None:
    plt = _ensure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: module-type histogram
    from collections import Counter

    types = Counter(layer.module_type for layer in record.layers)
    items = sorted(types.items(), key=lambda kv: -kv[1])[:25]
    if items:
        fig, ax = plt.subplots(figsize=(8, 6))
        labels, counts = zip(*items)
        ax.barh(range(len(labels)), counts)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("count")
        ax.set_title(
            f"{record.model_kind} module-type histogram [diagnostic-not-score]"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "module_type_histogram.png", dpi=120)
        plt.close(fig)

    # Figure 2: mixer-rank-vs-depth for attention-like layers
    attn = record.attention_layers()
    if attn:
        ranks = []
        names = []
        concs = []
        for layer in attn:
            af = layer.attention_fingerprint
            if af is None:
                continue
            r = af.mixer_rank_proxy if af.mixer_rank_proxy is not None else 0.0
            c = af.spatial_concentration if af.spatial_concentration is not None else 0.0
            ranks.append(r)
            concs.append(c)
            names.append(layer.name.replace("vision.stages.", "S").replace(".blocks.", ".B"))
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(range(len(ranks)), ranks, marker="o")
        # Use log-scale only when there are strictly positive values; SegNet's
        # smp.Attention=Identity blocks emit zero rank and would crash log.
        if any(r > 0 for r in ranks):
            axes[0].set_yscale("log")
        axes[0].set_ylabel("mixer rank proxy (log)" if any(r > 0 for r in ranks) else "mixer rank proxy")
        axes[0].set_title(
            f"{record.model_kind} mixer-rank vs depth (RepMixer) [diagnostic-not-score]"
        )
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(range(len(concs)), concs, marker="s", color="C1")
        axes[1].set_ylabel("spatial concentration\n(max(p))")
        axes[1].set_xlabel("block index")
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, fontsize=7, ha="right")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "mixer_rank_vs_depth.png", dpi=120)
        plt.close(fig)

    # Figure 3: activation numel distribution (output of each layer)
    numels = []
    for layer in record.layers:
        if layer.output_stats:
            numels.append(sum(s.numel for s in layer.output_stats))
    if numels:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(numels, bins=40)
        ax.set_xscale("symlog")
        ax.set_xlabel("activation numel (per layer)")
        ax.set_ylabel("count")
        ax.set_title(
            f"{record.model_kind} activation numel distribution [diagnostic-not-score]"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "activation_numel_histogram.png", dpi=120)
        plt.close(fig)


def _drift_figures(
    record_a: IntrospectionRecord,
    record_b: IntrospectionRecord,
    out_dir: Path,
) -> None:
    plt = _ensure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    drift = compute_layer_drift(record_a, record_b)

    # Figure: per-layer drift bar chart for the largest 30 entries.
    rows = []
    for name, entries in drift.items():
        for entry in entries:
            magnitude = (
                entry.l2_relative_error
                if entry.has_full_tensors
                else (entry.fingerprint_only_l2_proxy or 0.0)
            )
            rows.append((name, magnitude, entry))
    rows.sort(key=lambda r: r[1], reverse=True)
    rows = rows[:30]
    if rows:
        fig, ax = plt.subplots(figsize=(10, 8))
        names = [r[0] for r in rows]
        mags = [r[1] for r in rows]
        ax.barh(range(len(rows)), mags)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("L2 relative error / fingerprint proxy")
        ax.set_title(
            f"Per-layer drift: {record_a.device} vs {record_b.device} "
            f"[diagnostic-not-score]"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "per_layer_drift.png", dpi=120)
        plt.close(fig)

    # Figure: compounding factor along RepMixerBlock path.
    summary = estimate_compounding_for_path(drift)
    eps_seq = []
    block_names = []
    for name, entries in drift.items():
        if "vision.stages." in name and any(name.endswith(f".blocks.{i}") for i in range(20)):
            for entry in entries:
                eps_seq.append(entry.l2_relative_error)
                block_names.append(name.replace("vision.stages.", "S").replace(".blocks.", ".B"))
    if eps_seq:
        cum = []
        prod = 1.0
        for eps in eps_seq:
            prod *= 1.0 + eps
            cum.append(prod)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(cum)), cum, marker="o")
        ax.set_xticks(range(len(block_names)))
        ax.set_xticklabels(block_names, rotation=45, fontsize=8, ha="right")
        ax.set_ylabel("∏(1 + ε_i)")
        ax.set_title(
            f"Compounding factor along FastViT depth (final = "
            f"{summary['compound_factor_l2_rel']:.3f}) [diagnostic-not-score]"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "compounding_factor.png", dpi=120)
        plt.close(fig)

    (out_dir / "drift_summary.json").write_text(
        json.dumps(
            {
                "compounding_summary": summary,
                "tag": "[diagnostic-not-score]",
                "device_a": record_a.device,
                "device_b": record_b.device,
                "layers_compared": len(drift),
                "drift_rows": drift_to_dict(drift),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-a", type=Path, required=True)
    parser.add_argument("--record-b", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    record_a = IntrospectionRecord.from_disk(args.record_a)
    if args.record_b is None:
        _single_record_fingerprint_figures(record_a, args.output_dir)
        print(f"[viz] single-record figures -> {args.output_dir}")
    else:
        record_b = IntrospectionRecord.from_disk(args.record_b)
        _drift_figures(record_a, record_b, args.output_dir)
        print(f"[viz] drift figures -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
