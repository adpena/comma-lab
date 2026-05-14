"""Visualize the scorer-conditional MDL ablation bytemap.

Reads `<archive_name>_mdl_ablation.json` from Z1 ablation output and emits
a bytemap PNG showing per-byte-sample |Δscore_components| intensity by
archive offset.

Per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable
the output is human-readable; per "Forbidden /tmp paths" the PNG goes
under experiments/results/<lane>_<TS>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
except ImportError as e:
    print(f"[ERROR] matplotlib + numpy required: {e}", file=sys.stderr)
    raise


def render_bytemap(json_path: Path, output_path: Path) -> None:
    """Render bytemap from one archive's ablation JSON."""
    with open(json_path) as f:
        data = json.load(f)

    archive_name = data["archive_name"]
    archive_size = data["archive_size_bytes"]
    baseline_sc = data.get("baseline_score_components", 0.0)
    grammar = data.get("grammar", "unknown")
    device = data.get("device", "unknown")

    # Collect Tier B samples
    samples = []  # list of (byte_offset_global, abs_delta, inflate_success, section)
    for tb in data.get("tier_b", []):
        section_name = tb["section"]
        # Find section start offset from Tier A entries
        start = 0
        length = 0
        for ta in data.get("tier_a", []):
            if ta["section"] == section_name:
                start = ta["start_offset"]
                length = ta["length_bytes"]
                break
        for sample in tb.get("samples", []):
            offset_in_section = sample["byte_offset"]
            offset_global = start + offset_in_section
            dsc = sample.get("delta_score_components")
            inflate_ok = sample.get("inflate_success", False)
            abs_delta = abs(dsc) if dsc is not None else float("inf") if not inflate_ok else 0.0
            samples.append((offset_global, abs_delta, inflate_ok, section_name))

    if not samples:
        print(f"[WARN] no Tier B samples in {json_path}; skipping bytemap")
        return

    # Build figure: top panel = per-sample bar chart (offset vs |Δscore|);
    # bottom panel = per-section frac_significant histogram
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1.5])

    # Section colors
    sections_unique = []
    for tb in data.get("tier_b", []):
        if tb["section"] not in sections_unique:
            sections_unique.append(tb["section"])
    cmap = plt.get_cmap("tab10")
    section_color = {s: cmap(i % 10) for i, s in enumerate(sections_unique)}

    # Top: scatter of (byte_offset, abs_delta) colored by section
    # Cap inflate-failure inf at the max finite delta x 2 for plotting
    finite_deltas = [s[1] for s in samples if s[1] != float("inf")]
    max_finite = max(finite_deltas) if finite_deltas else 0.001
    cap = max_finite * 2.0 if max_finite > 0 else 0.01

    for section_name in sections_unique:
        xs = [s[0] for s in samples if s[3] == section_name]
        ys_raw = [s[1] for s in samples if s[3] == section_name]
        ys = [min(y, cap) if y != float("inf") else cap for y in ys_raw]
        # Inflate-failure markers
        fails = [(s[0], cap) for s in samples if s[3] == section_name and not s[2]]
        successes = [(s[0], min(s[1], cap)) for s in samples if s[3] == section_name and s[2]]
        if successes:
            xs_s, ys_s = zip(*successes)
            ax_top.scatter(xs_s, ys_s, c=[section_color[section_name]], s=12, alpha=0.6,
                           label=f"{section_name} (success)")
        if fails:
            xs_f, ys_f = zip(*fails)
            ax_top.scatter(xs_f, ys_f, c=[section_color[section_name]], s=20, alpha=0.9,
                           marker="x", label=f"{section_name} (inflate-fail)")

    # Annotate section boundaries with vertical lines
    for ta in data.get("tier_a", []):
        ax_top.axvline(x=ta["start_offset"], color="gray", linestyle=":", alpha=0.4)
        ax_top.axvline(x=ta["start_offset"] + ta["length_bytes"], color="gray", linestyle=":", alpha=0.4)

    ax_top.set_xlabel("Byte offset in archive")
    ax_top.set_ylabel("|Δscore_components| per byte flip")
    ax_top.set_title(
        f"Scorer-conditional MDL bytemap — {archive_name} ({archive_size:,} bytes, grammar={grammar})\n"
        f"baseline score_components = {baseline_sc:.4f}  device = {device}  inflate-failure capped at 2× max-finite Δ"
    )
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(alpha=0.3)

    # Bottom: per-section bar chart of frac_significant
    sec_names = [tb["section"] for tb in data.get("tier_b", [])]
    fracs = [tb.get("fraction_significant", 0.0) for tb in data.get("tier_b", [])]
    n_samples = [tb.get("n_samples", 0) for tb in data.get("tier_b", [])]
    section_lengths = []
    for sec_name in sec_names:
        for ta in data.get("tier_a", []):
            if ta["section"] == sec_name:
                section_lengths.append(ta["length_bytes"])
                break
        else:
            section_lengths.append(0)

    colors_bot = [section_color.get(s, "gray") for s in sec_names]
    bar_pos = np.arange(len(sec_names))
    bars = ax_bot.bar(bar_pos, fracs, color=colors_bot, alpha=0.7)
    ax_bot.set_xticks(bar_pos)
    ax_bot.set_xticklabels([f"{s}\n(L={l:,}B)" for s, l in zip(sec_names, section_lengths)], rotation=0, fontsize=9)
    ax_bot.set_ylabel("fraction significant\n(|Δ|>threshold OR inflate-fail)")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.axhline(y=1.0, color="black", linestyle=":", alpha=0.5)
    ax_bot.set_title(
        f"Per-section scorer-extraction fraction "
        f"(MDL_lo={data['mdl_scorer_extracted_bytes_lo']:.0f}B, "
        f"density={data['mdl_density_estimate_lo']:.3f})"
    )
    # Annotate bars with N and fraction
    for bar, frac, n in zip(bars, fracs, n_samples):
        ax_bot.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{frac:.2f} (N={n})", ha="center", fontsize=8)
    ax_bot.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.text(0.5, 0.005,
             f"Zen-floor band recommendation: {data.get('zen_floor_band_recommendation', 'N/A')}",
             ha="center", fontsize=9, style="italic")
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote bytemap: {output_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Visualize MDL ablation bytemap")
    p.add_argument("--input", type=Path, required=True, action="append",
                   help="MDL ablation JSON path (one per archive; repeat)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for json_path in args.input:
        out_name = json_path.stem + "_bytemap.png"
        out_path = args.output_dir / out_name
        render_bytemap(json_path, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
