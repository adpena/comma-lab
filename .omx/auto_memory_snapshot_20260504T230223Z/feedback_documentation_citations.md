---
name: Documentation with Citations — Paper and OSS References
description: Code must be documented with citations to papers and OSS where appropriate. Journal-grade documentation.
type: feedback
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
All code must be documented with citations to relevant papers and OSS projects where appropriate.

**Why:** This is a research codebase destined for arXiv publication and open-source release. Every non-obvious technique should cite its source. This is what journal-grade code looks like — it connects implementation to theory.

**How to apply:**
- Cite papers in docstrings when implementing a published technique (e.g., "Lagrangian annealing per [Bertsekas 1982]")
- Cite OSS when using or adapting someone's implementation (e.g., "Adapted from torchvision.models.optical_flow.raft_small")
- Use standard citation format: [Author Year] or full BibTeX-style in module docstring
- Link to repos/papers with URLs where helpful
- Don't over-cite obvious things (no citation needed for "we use Adam optimizer")
- DO cite: loss function designs, architectural choices, optimization strategies, scoring formula derivations
- The test: could a reader trace every non-trivial technique back to its source?
