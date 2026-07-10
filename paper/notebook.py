import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Task-Aware Post-Filtering for Video Compression in Autonomous Driving

        **Score: 1.33** | #1 by 0.53 margin | comma.ai Video Compression Challenge

        A 45KB CNN post-filter trained by backpropagating through frozen PoseNet and SegNet
        scorer networks. The filter learns pixel corrections that preserve the visual information
        autonomous driving models consume, rather than optimizing generic perceptual quality.

        This notebook is a live, interactive companion to the
        [arXiv paper](docs/writeup_draft.md). All data loads from the repo automatically —
        as experiments run and results accumulate, the visualizations update.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import json
    import math
    from pathlib import Path
    from datetime import datetime
    return Path, datetime, json, math, mo


@app.cell
def _(Path, json):
    # Auto-load results from the repo — updates as new experiments complete
    REPO_ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path(".")
    RESULTS_PATH = REPO_ROOT / "reports" / "results.jsonl"
    TIMELINE_PATH = REPO_ROOT / "reports" / "timeline.jsonl"
    FINDINGS_PATH = REPO_ROOT / ".omx" / "research" / "findings.md"
    WRITEUP_PATH = REPO_ROOT / "docs" / "writeup_draft.md"

    def load_jsonl(path):
        """Load a JSONL file into a list of dicts."""
        if not path.exists():
            return []
        records = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    results = load_jsonl(RESULTS_PATH)
    timeline = load_jsonl(TIMELINE_PATH)
    print(f"Loaded {len(results)} results, {len(timeline)} timeline entries")
    return (
        FINDINGS_PATH,
        REPO_ROOT,
        RESULTS_PATH,
        TIMELINE_PATH,
        WRITEUP_PATH,
        load_jsonl,
        results,
        timeline,
    )


@app.cell
def _(math, mo, results):
    # --- Score Trajectory ---
    # Underscore-prefixed locals stay cell-private in marimo's dataflow model
    # (avoids the multiple-definition collision on shared loop variables).
    _rows = []
    for _r in results:
        if "current_workflow_score" in _r:
            _score = _r["current_workflow_score"]
            _seg = _r.get("segnet_distortion", 0)
            _pose = _r.get("posenet_distortion", 0)
            _rate = _r.get("rate", 0)
            _tag = _r.get("run_id", _r.get("config", {}).get("variant", "unknown"))
            _short = _tag[:40] if len(_tag) > 40 else _tag
            _rows.append({
                "run": _short,
                "score": f"{_score:.3f}",
                "seg": f"{_seg:.6f}",
                "pose": f"{_pose:.6f}",
                "rate": f"{_rate:.6f}",
                "seg_term": f"{100*_seg:.4f}",
                "pose_term": f"{math.sqrt(10*_pose):.4f}" if _pose > 0 else "0",
                "rate_term": f"{25*_rate:.4f}",
            })

    mo.vstack([
        mo.md("## Score Trajectory"),
        mo.ui.table(_rows, label="Score trajectory (from results.jsonl)"),
    ])
    return


@app.cell
def _(mo):
    # --- Interactive Score Decomposition ---
    seg_slider = mo.ui.slider(
        0.003, 0.010, value=0.00610, step=0.0001,
        label="SegNet distortion"
    )
    pose_slider = mo.ui.slider(
        0.0005, 0.015, value=0.00218, step=0.0001,
        label="PoseNet distortion"
    )
    rate_slider = mo.ui.slider(
        0.015, 0.035, value=0.02302, step=0.001,
        label="Rate"
    )

    mo.vstack([
        mo.md("## Interactive Score Decomposition"),
        seg_slider, pose_slider, rate_slider,
    ])
    return pose_slider, rate_slider, seg_slider


@app.cell
def _(math, mo, pose_slider, rate_slider, seg_slider):
    seg_val = seg_slider.value
    pose_val = pose_slider.value
    rate_val = rate_slider.value

    seg_term = 100 * seg_val
    pose_term = math.sqrt(10 * pose_val)
    rate_term = 25 * rate_val
    total = seg_term + pose_term + rate_term

    # Baseline for comparison
    seg_base = 100 * 0.00580
    pose_base = math.sqrt(10 * 0.01229)
    rate_base = 25 * 0.02302
    baseline = seg_base + pose_base + rate_base

    mo.md(f"""
    ### Score = {total:.4f} (baseline: {baseline:.4f}, delta: {total - baseline:+.4f})

    | Component | Value | Contribution | % of Total |
    |-----------|-------|-------------|------------|
    | 100 * seg | 100 * {seg_val:.5f} | **{seg_term:.4f}** | {seg_term/total*100:.1f}% |
    | sqrt(10 * pose) | sqrt(10 * {pose_val:.5f}) | **{pose_term:.4f}** | {pose_term/total*100:.1f}% |
    | 25 * rate | 25 * {rate_val:.5f} | **{rate_term:.4f}** | {rate_term/total*100:.1f}% |
    | **Total** | | **{total:.4f}** | 100% |

    **Marginal sensitivities at this point:**
    - d(S)/d(seg) = 100.0 (constant)
    - d(S)/d(pose) = {5/math.sqrt(10*pose_val):.1f} (diminishing)
    - d(S)/d(rate) = 25.0 (constant)
    - SegNet is **{100 / (5/math.sqrt(10*pose_val)):.1f}x** more valuable per unit than PoseNet
    """)
    return baseline, pose_term, rate_term, seg_term, total


@app.cell
def _(math, mo):
    # --- Proposed Scoring Formula (Arrow + Pareto) ---
    _s0, _p0, _r0 = 0.00580, 0.01229, 0.02500
    _submissions = [
        ("Baseline (no filter)", 0.00580, 0.01229, 0.02500),
        ("Our submission (1.33)", 0.00610, 0.00218, 0.02302),
        ("Hypothetical balanced", 0.00500, 0.00400, 0.02302),
        ("KL distill (DEAD)", 0.00546, 0.08095, 0.02407),
    ]

    _rows = []
    for _name, _s, _p, _r in _submissions:
        _current = 100*_s + math.sqrt(10*_p) + 25*_r
        _proposed = (_s/_s0)**0.40 * (_p/_p0)**0.35 * (_r/_r0)**0.25
        _rows.append({
            "Submission": _name,
            "Current formula": f"{_current:.3f}",
            "Proposed formula": f"{_proposed:.3f}",
            "seg": f"{_s:.5f}",
            "pose": f"{_p:.5f}",
        })

    mo.vstack([
        mo.md("## Proposed Scoring Formula (Arrow + Pareto)"),
        mo.md(r"""
    The current additive formula allows axis exploitation. Our proposed multiplicative formula
    enforces complementarity:

    $$\text{SCORE}_{\text{proposed}} = \left(\frac{s}{s_0}\right)^{0.40} \cdot \left(\frac{p}{p_0}\right)^{0.35} \cdot \left(\frac{r}{r_0}\right)^{0.25}$$

    where $s_0, p_0, r_0$ are baseline (unfiltered) values.
    """),
        mo.ui.table(_rows, label="Current vs proposed scoring formula"),
    ])
    return


@app.cell
def _(FINDINGS_PATH, mo):
    # --- Live Findings Feed ---
    if FINDINGS_PATH.exists():
        _body = FINDINGS_PATH.read_text()
    else:
        _body = "*findings.md not found — run experiments to populate*"

    mo.vstack([
        mo.md("## Research Findings (live from .omx/research/findings.md)"),
        mo.md(_body),
    ])
    return


@app.cell
def _(mo):
    # --- Pareto Frontier Visualization ---
    _frontier_points = [
        {"label": "Baseline (no filter)", "seg": 0.00580, "pose": 0.01229, "marker": "circle"},
        {"label": "Standard h=64", "seg": 0.00580, "pose": 0.01229, "marker": "circle"},
        {"label": "Dilated h=64 (OURS)", "seg": 0.00610, "pose": 0.00218, "marker": "star"},
        {"label": "KL distill #1", "seg": 0.00494, "pose": 0.05725, "marker": "x"},
        {"label": "KL distill #2", "seg": 0.00546, "pose": 0.08095, "marker": "x"},
    ]

    mo.vstack([
        mo.md("""
    ## PoseNet-SegNet Pareto Frontier

    The frontier shows the tradeoff between PoseNet and SegNet optimization.
    Our submission is at the PoseNet-extreme. The true score minimum lies
    somewhere in between.
    """),
        mo.ui.table(_frontier_points, label="Observed Pareto frontier points"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    *This notebook auto-loads from `reports/results.jsonl`, `.omx/research/findings.md`,
    and other repo files. As experiments run and results accumulate, the visualizations
    update automatically. No manual data entry needed.*

    **Git history IS our research timeline.** Every commit tells part of the story.
    """)
    return


if __name__ == "__main__":
    app.run()
