# CLAUDE.md Non-Negotiable Subsection Proposal — ARBITRARINESS-EXTINCTION META-LENS

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Status**: PROPOSAL — awaits operator decision
**Captured at UTC**: `2026-05-18T21:20:00Z`
**Source**: operator canonical naming 2026-05-18 verbatim

## Operator standing directive 2026-05-18 (verbatim)

> "The concept I see is identifying arbitrariness or less than optimal being applied across the board and using all techniques and exploits and contest rules and allowed and everything to either experimentally determine the proper solution or solve it and use the optimal solution or use a formula instead or learn and train against the values or use neural or self or some other alien tech or combination of teks"

## Proposed CLAUDE.md subsection (insert near "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS")

```markdown
## ARBITRARINESS-EXTINCTION META-LENS — NON-NEGOTIABLE, HIGHEST EMPHASIS

**Source:** operator canonical naming 2026-05-18 verbatim (above). Anchor memo:
`feedback_arbitrariness_extinction_meta_lens_systematic_audit_landed_20260518.md`.
Sister of CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" + Catalog #303
(cargo-cult audit per substrate) + Catalog #229 (premise verification before edit) +
HNeRV parity discipline L6 (score-domain Lagrangian not weight-domain proxies).

### The 5-path resolution framework (binding)

Every arbitrary value in the contest stack — hyperparameter, default, threshold,
multiplier, grid choice — MUST be resolved via ONE of:

1. **EXPERIMENTAL** — sweep + empirical optimum (cheap; LR finder, K sweep, validation-set
   probe). Use when no closed-form available.
2. **ANALYTICAL SOLVE** — KKT / convex optimization / closed-form math (water-filling,
   FISTA, mirror descent, Lagrange-multiplier derivation). Use when objective is convex
   or admits closed-form solution per Boyd-Vandenberghe / Bertsekas.
3. **FORMULA** — Shannon R(D) / MDL / first-principles bound (Hessian-trace importance,
   -log p(y) bits, Polyak-Juditsky averaging window, Smith cyclical LR). Use when
   information-theoretic or statistical-physics derivation exists.
4. **LEARNED / TRAINED** — neural surrogate or end-to-end SGD (Ballé hyperprior,
   Quantizr renderer, Hinton distillation, Kendall uncertainty-weighted multi-task loss).
   Use when the optimum is data-dependent and not closed-form.
5. **SELF / ALIEN-TECH composition** — self-compression / cross-paradigm / VQ-VAE
   K-means++ init / Riemannian-Newton / Mamba SSM / DreamerV3 RSSM / DP1 codebook
   composition. Use when no canonical optimum exists and breakthrough requires
   cross-disciplinary primitive.

### Mandatory per-substrate-design-memo section

Every NEW substrate design memo at `.omx/research/*_design_<YYYYMMDD>.md` dated
>= 2026-05-18 MUST include a section `## Arbitrariness audit per layer` that:

1. **Enumerates each per-layer arbitrary value** introduced by the substrate
   (learning rate, EMA decay, batch size, codebook K, threshold values, etc.)
2. **Per-value resolution-path classification** per the 5 paths above
3. **Per-value EV envelope + cost envelope** for the resolution
4. **Per-value canonical-helper citation** OR explicit `arbitrariness_inherited_from_canonical`
   with rationale (sister to Catalog #290 canonical-vs-unique decision)
5. **Per-value literature citation** (no path-of-least-resistance allowed without
   canonical citation)

The literal section header is the single structural requirement; the section body
content is the operator-facing audit surface. Same-line waiver
`# ARBITRARINESS_AUDIT_SECTION_WAIVED:<rationale>` accepted (placeholder
`<rationale>` / `<reason>` rejected).

### Forbidden patterns

1. **Cargo-culted defaults** — copying `lr=5e-4` / `ema_decay=0.997` / `batch_size=8` /
   `K=64` / `threshold=0.5` from sister substrate without per-method-optimal evaluation
   per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode". The new substrate's
   optimal value may differ.
2. **Hardcoded grids** — `_ALLOWED_QINT_MAX = (1, 3, 7, 15, 31)` or similar fixed
   discrete grids without justification that the continuous solution is unavailable
   or worse.
3. **Hand-tuned multipliers without canonical derivation** — `λ_seg`, `λ_pose`, `λ_rate`
   in score-aware loss must derive from the contest formula's analytic derivatives
   per Catalog #322 cascade-2 + operating-point-dependent rule, NOT hand-tuned.
4. **Single seed per run** — per Reimers-Gurevych 2017 arxiv:1707.09861, seed
   variance can dominate technique differences. Use k=3 seed ensemble + take min
   for any score-claim run.
5. **Default activation / init undocumented** — every architecture must justify
   activation function + init scheme per Saxe et al 2014 arxiv:1312.6120.

### Enforcement

Proposed STRICT preflight Catalog #~333
`check_substrate_design_memo_has_arbitrariness_audit_section` refuses post-2026-05-18
substrate design memos lacking the canonical section. Initial wire-in is
WARN-ONLY per "Strict-flip atomicity rule"; strict-flip after operator-routed
backfill of in-flight design memos.

Companion canonical helper `tac.arbitrariness_extinction_lens` (proposal at
`.omx/research/canonical_helper_proposal_tac_arbitrariness_extinction_lens_20260518.md`)
automates the per-value AST scan + 5-path classifier + ranker + autopilot wire-in.

### Empirical proof of concept

The just-landed MORE-OPTIMAL ALGORITHMS subagent (commit `35c5d429f`) PROVED the lens
applied to SOLVER SELECTION yields 3 empirical wins in 30 min for $0:

* FISTA empirically beat water-filling bisection 1.25× with byte-identical solution
* Frank-Wolfe empirically beat Sinkhorn 1.9× (FALSIFYING the predicted "Sinkhorn-wins"
  prior in the synthesis memo)
* Riemannian-Newton on Stiefel beat Lloyd-projection 1.88× at machine-ε orthogonality

The lens applied across the full contest stack (training-time + codec-time +
inflate-time + composition + council surfaces) yields the canonical 52-row inventory
at `.omx/state/arbitrariness_extinction_audit_20260518.jsonl` with TOTAL predicted ΔS
envelope **[-0.139, -0.026]** at total cost ~$120 (31 of 52 rows at $0 cost).

### Cross-references

- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" — the parent
  variational principle this lens operationalizes
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — sister rule:
  per-method optimization includes per-method de-arbitrariness
- Catalog #303 (cargo-cult audit per substrate) — design-memo surface; sister
- Catalog #229 (premise-verification-before-edit) — per-edit surface; sister
- HNeRV parity discipline L6 (score-domain Lagrangian) — the canonical λ
  derivation surface this lens enforces per-substrate
- Catalog #322 (composition α v2 cascade) — composition-time arbitrariness
  surface; consumes lens output via cathedral autopilot ranker
```

## Operator decision points

1. **Greenlight the CLAUDE.md amendment as written?** (Insert as proposed subsection)
2. **Greenlight Catalog #~333 STRICT preflight gate?** (Sister to Catalog #303 + #305; initial WARN-ONLY)
3. **Greenlight `tac.arbitrariness_extinction_lens` canonical helper build?** (Per sister proposal memo)
4. **Greenlight backfill sweep of in-flight substrate design memos?** (Multi-subagent backfill wave; adds `## Arbitrariness audit per layer` section to existing 2026-05-18 design memos)

## Cost envelope for full operator approval

- CLAUDE.md amendment: 1 commit, $0
- Catalog #~333 STRICT gate: ~80 LOC + ~15 tests, $0
- `tac.arbitrariness_extinction_lens` module: ~600 LOC + ~30 tests, $0 GPU, ~1.5h
- Backfill sweep of existing design memos: ~10 subagent slots, $0 GPU
- Cathedral autopilot wire-in: ~20 LOC, $0
- **TOTAL: $0 GPU, ~3h total subagent wall-clock**
