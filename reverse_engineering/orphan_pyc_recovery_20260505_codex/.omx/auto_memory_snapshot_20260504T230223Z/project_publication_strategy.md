---
name: Two-Tier Publication Strategy
description: writeup_draft.md = arXiv paper, Cloudflare site = accessible landing page with dynamic viz
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Decision (2026-04-10)

Two publication surfaces, each with different audiences and formats:

### 1. arXiv Paper (`docs/writeup_draft.md`)
- Formal academic paper (Nature/JMLR style)
- Static figures (PNG/PDF for LaTeX)
- Full citations, proofs, theoretical framework
- 7 sections: Intro, Background, Method, Negative Results, Multi-Objective Analysis, Recommendations, Results
- Competition-sensitive details delayed until after results
- Tone: precise, evidence-driven, understated

### 2. Cloudflare Site (reports/graphs/site/)
- Accessible landing page for general audience
- Dynamic GIFs (comparison, color-coded SegNet diff)
- Interactive visualizations (Pareto frontier explorer, score decomposition)
- Narrative storytelling — the journey, not just the results
- Can be more specific and detailed (user explicitly approved)
- Link confined to private repo docs until user says to share broadly

### Shared content
- Static versions of GIFs go in the arXiv paper as figures
- The site links to the arXiv paper for formal details
- Both reference the same score numbers and experimental results

**Why:** Different audiences need different formats. Reviewers want rigor. Engineers want to see the GIF.
**How to apply:** When generating visualizations, always produce both a static high-res version (for paper) and an animated/interactive version (for site).
