# ddm_gdl1 — operator drop arXiv 2104.13478: Geometric Deep Learning blueprint crosswalk

**Drop (2026-08-07):** https://arxiv.org/pdf/2104.13478 — Bronstein, Bruna, Cohen, Veličković,
"Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" (the GDL proto-book).
Foundational survey (~150pp), not a technique paper — so the crosswalk is LENS-shaped: where
does the symmetry/invariance/equivariance blueprint predict a lever or explain a measured fact
on OUR frozen-scorer campaign? Rigor-triage is about APPLICABILITY (the math is settled), and
per the fractal-audit standard: read the RELEVANT chapters deeply, not a skim of all 150pp.

**Seeded crosswalk surfaces (adjudicate these AND sweep beyond):**
1. **Equivariance-breaking = our measured lattice facts.** The scorer's stride-2 stem breaks
   translation equivariance to the even-shift subgroup — m86 (disjoint 2×2 sampling, 4 private
   camera px/scorer px, 22.70% both-scorer-blind, lattice decouples EXACTLY) is this theorem
   measured. Question the lens raises: is the QUOTIENT group (shifts mod 2, the phase coset)
   an unexploited coordinate? Named consumers: #920 Road↔Lane PHASE positional DOF (0.110 S) ·
   the token-grid alignment choice in the semantic vehicle · pixel-phase placement (#149 lineage).
2. **Groups/geodesics = the ξ/se(3) line.** tac.lie, Chasles screw, warp-pose6 basis — already
   embodied. Check for MISSING pieces the blueprint names (e.g., proper group-convolution over
   SE(3) cosets vs our per-pair twist; canonicalization-vs-equivariance tradeoffs for the pose
   stream candidates in the pk1 race).
3. **Gauge chapter vs our gauge language.** ker(A) 80.67% nullity DOF / RANGE_A_COMPLEMENT ~52%
   / gauge-fixing (#519/#580) — does the blueprint's gauge-field formalism sharpen the fiber→
   gauge conversion ladder (#669a) or name a canonical gauge choice we lack?
4. **Graphs = the per-edge gap graph.** m91 (Road hub 87.8% of flips, Road↔Lane = 49.2%) is a
   graph-structured error object; the blueprint's message-passing lens vs our per-edge
   decomposition doctrine — is there an ADOPT-CLASS for edge-conditioned correction carriers?
5. **Equivariant/steerable architectures for the renderer fine-tune** — likely LESSON-ONLY
   (vehicle inherited, symmetry of the TASK is broken by the scorer anyway), but adjudicate
   honestly: does a scorer-symmetry-matched architecture class beat plain conv at matched bytes?
   (This is a HYBRID row for ty1/#979's synergy table, not a build order.)

**Deliverable:** ranked crosswalk table {ADOPT / ADOPT-CLASS / LESSON-ONLY / ALREADY-EMBODIED /
N-A} with named consumers per row (#984 composition · pk1 pose race · #920/#149 phase levers ·
#669a gauge ladder · ty1 synergy rows), honesty labels (MEASURED/DERIVED/CONJECTURE), and for
any ADOPT: the smallest $0 falsifiable probe, pre-registered. Full research authority (online +
OSS + corpus) per standing arm contract. De-dupe against prior crosswalks (cl1 Clifford/GA ·
#464 Weyl symmetry · #468 Lie-groups dig · #476 geometry-per-slot · #552 SPD momentum) — cite,
don't re-derive; NEW rows only.

**Boundaries:** CPU-only, NO Metal, no scorer slot. Findings:
`.omx/research/ddm_gdl1_20260807/GDL1_CROSSWALK.md` + typed JSONL rows.

**Discipline:** serializer + POST-EDIT `--expected-content-sha256` per file; tags
`[no-triality] [p0-ledger-ok]`; review_tracker ×2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer. If serializer hits sandbox git-perms, write artifacts + say so.
