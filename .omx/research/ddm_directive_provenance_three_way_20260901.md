# OPERATOR DIRECTIVE 2026-09-01 — THREE-WAY PROVENANCE ACCOUNTING (binding on ddm_pq13; MAIN enforces at harvest; bears on sp978's verdict framing and the #1156 writeup)

**Operator verbatim:** "We also want to be clear about what is ours and what is theirs and
what we had been working on independently but hadn't submitted yet when PR130 and 135 hit."

**Binding consequence:** the PR body's NO-FAKE #7 `borrowed_substrate_accounting` section is
upgraded from a two-way to a THREE-WAY table. Every mechanism/component in the shipping
packet (and every headline claim in the draft) is classified into exactly one of:

1. **THEIRS (borrowed, credited):** adopted from PR130/PR135 (or other public PRs), named
   with the PR number and author. Known members: the HPAC integer coder lineage · the
   CPR1 container · the semantic-pose mechanism as SHIPPED by PR130/135 · the RC64 stream
   format our frontier body descends from. Plain language, no hedging — the frontier BODY is
   a PR130-lineage recode; crediting it fully is what makes the OURS columns credible.
2. **OURS (original):** mechanisms with no borrowed ancestor. Known members: the 23 pointer-
   move mechanisms (address-free tile48 × groupbin8, the micro-edit/realized-acceptance
   engine, Schur in-compile compensation, the tail-override law, fixed-point log-odds mixer)
   · the analysis/law layer (conditioning-transport, sharp-optimum, the Cross, round-trip
   affine) · the witness/level-set solver stack (ker(A)/resize-nullity projector, preimage
   compiler, exact composite-R adjoint, se(3) engine) · the seal/fire/identity custody
   apparatus.
3. **OURS-INDEPENDENT, PRE-DATING, UNSUBMITTED:** work in OUR tree with commit dates BEFORE
   the respective PR landed publicly, converging on ideas those PRs later shipped — honest
   independent-invention claims, never priority claims over their PUBLICATION. Seed rows the
   table must verify AT SOURCE (git log --follow, commit hash + author date per row; the PR
   publish dates fetched from GitHub createdAt, not from memory):
   - **Partition-as-primary-object** (the task-space/level-set witness capstone — the
     partition itself as the carried object): operator directive 2026-06-25 + the
     SPEC/trainer commits of late June — ~a month before PR130's dense-semantic-token
     mechanism reached us (pi1 intake 2026-07-27). PR130 shipped a semantic-token vehicle
     publicly first; we had the semantic-object PREMISE in-tree first. State both halves.
   - **Stored pose-target sidecar** (`src/tac/scorer_targets.py`, 600×6 fp16 + FiLM
     conditioning design): June — vs PR130/135's shipped semantic-pose carrier.
   - **Boundary-math / direct-partition stack** (`src/tac/boundary_math/*`, contour/RAG/
     region-merge/margin-polytope): June.
   - **Context/arithmetic coder program** (coder races, SMEVR, ANS/range corpus): July,
     partially prior, partially parallel — date each honestly; where PR130's HPAC predates
     a specific member of ours, that member goes in column 1 or gets a dual-dated row.
   - Any further rows the census surfaces — derived from receipts, never padded.

## Rules (binding)

- **Receipts, not memory:** every row carries {our first in-tree commit hash + author date ·
  the PR's public createdAt · one-line mechanism description}. A row that cannot pin both
  dates is dropped or moved to an honest "undated" note — never asserted (m153 /
  #1357 genera).
- **Honesty is symmetric:** the third column claims INDEPENDENT CONCEPTION with evidence; it
  never claims publication priority, and it never dilutes column 1's credit. Over-claiming
  here is a NO-FAKE #7 violation exactly as much as under-crediting them.
- **Where it lands:** (a) the PR_BODY_DRAFT v2's accounting section (pq13); (b) a standalone
  committed provenance table other consumers cite (the #1156 writeup, sp978's verdict
  framing — sp978 should describe the semantic-primary treatment as "PR130/135's shipped
  mechanism × our pre-dated semantic-object line", with the dates); (c) the operator
  sign-off summary highlights the three-way split explicitly.
- **Enforcement:** pq13 may have started before this directive landed — MAIN verifies the
  three-way table at pq13's harvest and supplies the git-dated derivation itself if the arm
  missed the amendment; the draft does NOT go to the operator without it.

## Provenance

Fifth operator steer of 2026-09-01. Anchors: NO-FAKE #7 (borrowed-substrate accounting) ·
m07 SHIP-LAW honesty-half · pq1's original accounting table (the object being upgraded) ·
pi1 (#728) PR130 intake 2026-07-27 · #1009 PR135 intake 2026-08-10 · afr1 archive SHA
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
