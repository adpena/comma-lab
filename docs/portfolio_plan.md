# Portfolio Page Plan: Comma.ai Video Compression Challenge

## Hero Section

**Headline:** "Ranked #1 in the comma.ai Lossy Video Compression Challenge"

**Subline:** Score 1.727 — a composite metric combining bitrate, segmentation accuracy, and pose estimation fidelity across 1200 frames of autonomous driving footage.

**Key stats row (3-4 pills):**
- Score: 1.727 (1st place)
- Experiments: 18 controlled runs
- Open-source library: `tac v0.5.0` (7 modules)
- Research log: 50+ entries over multi-week sprint

---

## Technical Summary (3-4 paragraphs)

**Paragraph 1 — The problem.**
Frame the challenge: compress 1200 frames of driving video so that downstream perception models (SegNet, PoseNet) still perform well on the reconstructed output. The scoring metric is a weighted combination of bitrate, segmentation error, and pose estimation error. Naive codec tuning hits a wall fast because PoseNet is extraordinarily sensitive to any pixel-level degradation.

**Paragraph 2 — The key insight.**
Describe the discovery that PoseNet's Jacobian is near-rank-1 with a linear radius under 0.0001 px, meaning closed-form correction methods are useless and any preprocessing (blur, chroma subsampling, even gentle sharpening) is fatal. The only viable path is encoder-parameter optimization combined with ROI-aware bitrate allocation that protects perception-critical regions while aggressively compressing sky and road surfaces.

**Paragraph 3 — The system.**
Outline the final pipeline: AV1 encoding with tuned CRF, ROI delta-q maps derived from semantic segmentation, quantization-aware training with EMA smoothing, and a lightweight 7.5 KB post-filter CNN trained against scorer gradients. Mention the trajectory from 2.08 to 1.845 to 1.727 and the dead ends that shaped the path (all preprocessing fails, post-filter distribution shift, Newton-method Jacobian collapse).

**Paragraph 4 — The engineering.**
Briefly cover the infrastructure: fleet orchestration across 4 machines (Mac, CUDA server, Mac Mini, tertiary), a custom Swift menubar monitoring app, 10 rounds of automated code review with 5/5 clean passes, and the `tac` library extracted from the work as a reusable pip-installable toolkit.

---

## Key Visualizations to Embed

Pick 3-4 from the existing D3 interactive site. Each should have a static fallback image for the portfolio page plus a "view interactive version" link.

1. **Score trajectory chart** — X-axis: experiment number, Y-axis: composite score. Annotate the key inflection points (ROI maps, post-filter, QAT+EMA). This is the hero visual.
2. **Pareto frontier** — Bitrate vs. perception quality tradeoff, showing where the final submission sits relative to naive baselines and competitor entries.
3. **ROI map overlay** — Side-by-side: original frame, semantic segmentation, delta-q ROI map, reconstructed frame. Shows the "why" behind the bitrate allocation.
4. **Dead-end summary table** — Compact grid of negative results (preprocessing, Newton, naive post-filter) with a one-liner on why each failed. Demonstrates rigor and honesty.

---

## Links to Artifacts

| Artifact | Link target | Notes |
|---|---|---|
| GitHub repo | `github.com/adpena/comma-lab` | Point to README + `tac/` library |
| Interactive writeup | Cloudflare site | Full D3 visualizations, experiment details |
| Technical paper | PDF or site subpage | The 18-experiment writeup |
| `tac` library | PyPI or GitHub subdir | 7 modules, v0.5.0, certified via review |
| Research diary | Repo or site subpage | 50+ entries, optional deep link |

---

## What This Demonstrates About the Author

Target these five competency signals — each maps to what top AV/ML teams care about:

1. **ML engineering depth.** Not just training models — understanding scorer internals, Jacobian structure, sensitivity analysis, and why naive approaches fail. This is the difference between "I trained a model" and "I understood the optimization landscape."

2. **Systems and infrastructure.** Fleet orchestration, CI-style review loops, artifact packaging, deployment to Cloudflare. End-to-end ownership from research to production-quality code.

3. **Research methodology.** 18 structured experiments with controls. Negative results recorded and learned from. Expert council consultation process (Tao, Karpathy, LeCun perspectives synthesized). AI-assisted but human-directed.

4. **Software quality.** `tac` library extracted as reusable open-source work. 10 rounds of code review, all clean. Type-checked, tested, documented. Not a one-off competition hack.

5. **Communication.** Interactive writeup with D3 visualizations. Research diary with 50+ entries. 1,360-event conversation timeline. The ability to explain complex tradeoffs clearly.

---

## Page Structure and Tone

- **Length:** Single scroll, 800-1000 words max on the portfolio page itself. Everything else lives behind links.
- **Tone:** Confident but precise. Lead with results, back with methodology, close with what it says about how you work. No hype words ("groundbreaking", "revolutionary"). Let the #1 rank speak.
- **CTA:** "Read the full writeup" (link to interactive site) and "View the code" (link to GitHub).
- **Placement:** Feature as the lead project on the portfolio. Pin it. This is the strongest signal piece.

---

## Implementation Notes

- Static site generator (Hugo, Astro, or plain HTML) is fine. The portfolio page itself should load fast.
- Embed static chart images with links to the interactive versions on the Cloudflare site. Do not bloat the portfolio with D3 dependencies.
- Keep the GitHub repo clean before linking: ensure README is polished, `tac` has a proper package structure, and the submission directory is tidy.
- Consider a 30-second screen recording of the interactive site as a fallback for recruiters who will not click through.
