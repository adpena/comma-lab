# Breakout Scaffold Landing And Meta-Assumption Guard - 2026-05-15

## Purpose

Record the post-interruption landing that moved the PR95/meta-level lesson from
chat into source, tests, recipes, and durable research state without promoting
unproven scaffold scores.

## Source commits

- `0b5c55538` on `main`: research-only breakout scaffold batch for ATW,
  NSCS01, NSCS02, NSCS03, Z3G2/Wunderkind, Catalog #290 extension, fail-closed
  recipes, and local verification.
- `9c9e17d18` local follow-up: Catalog #291
  `check_session_has_recent_meta_assumption_review`, a recurring
  META-ASSUMPTION ADVERSARIAL REVIEW cadence guard. Awaiting push with the
  post-review hardening patch below.

## Evidence landed in `0b5c55538`

- ATW, NSCS01, NSCS02, NSCS03, and Z3G2 are code-present and test-covered, but
  intentionally fail-closed as research/smoke surfaces until paired CPU+CUDA
  exact-eval custody exists.
- Z3G2 byte-mutation verifier now distinguishes semantic output mutation from
  parser-bound rejection. Parser rejection is consumption evidence, not score
  evidence.
- Catalog #290 now scans `.omx/research/*_design_*.md` memos and requires the
  `## Canonical-vs-unique decision per layer` section so design memos cannot
  silently force-share canonical helpers.
- `AGENTS.md` now requires obvious-fit proof before adopting canonical helpers;
  unclear cases fork or run paired smoke instead of defaulting to canonical.

## Verification

- `243 passed` for ATW, NSCS01, NSCS02, NSCS03, Z3G2, and Catalog #290 focused
  tests.
- `26 passed` for Catalog #290 + #291 focused tests after the #291 follow-up.
- `ruff` passed on all new scaffold packages, new tests, and the Z3G2 mutation
  verifier. `preflight.py` still carries unrelated legacy F841 debt, so the
  follow-up used targeted `F821,F401` safety checks there.
- `py_compile` passed for new trainers and submission Python runtimes.
- `bash -n` passed for new remote drivers and submission `inflate.sh` files.
- `git diff --check` passed.
- Live #291 operator-memory check returned `violations=0`.

## Post-review hardening patch

Two xhigh read-only reviewers returned high-confidence findings after the
`0b5c55538` push. The follow-up patch:

- Downgrades ATW, NSCS01, NSCS02, NSCS03, Z3G2, STC-Dasher, and U-DIE-KL
  overclaims from score bands to probe-gated hypotheses where evidence was
  not yet sufficient.
- Marks the assumptions matrix stack rows as interaction-probe hypotheses, not
  additive score predictions, and downgrades autopilot hooks to design-only
  until real `CandidateRow`/ranking inputs exist.
- Adds package-side `SubstrateContract` registration for ATW, NSCS02, and
  NSCS03, teaches Catalog #241 to accept package-side `registered_substrate`
  imports, and teaches Catalog #242 to import all substrate registration files.
- Aligns NSCS01 recipe/contract hardware to T4 and adds the missing head0 probe
  plan tool.
- Reclassifies the Z3G2 mutation verifier as parser/intermediate evidence, not
  full-frame `inflate.sh` output evidence.
- Hardens NSCS03 archive loading with `weights_only=True` and strict
  `str -> Tensor` validation; the standalone submission shim now deserializes
  all state blobs and remains research-only until byte-equivalent with the
  package runtime.
- Sets U-DIE-KL `die_cache_interval` default to 1 so cache correctness is exact
  unless a trainer opts into deterministic keyed caching.
- Applies the second reviewer wave:
  - Z3G2 remains `research_only=true` / `dispatch_enabled=false`; CPU/CUDA
    bands are null until full-frame mutation proof and paired exact eval exist.
  - NSCS01 `_full_main` and docs no longer claim implementation readiness while
    the path raises `NotImplementedError`.
  - NSCS02 recipe is unranked pending the resizing-chain ablation; missing
    remote-driver references were removed.
  - Catalog #290's CLAUDE.md row now matches the implementation: repo-local
    `.omx/research/*_design_*.md` by default, external Claude memory only by
    explicit opt-in.
  - Z3G2 verifier artifacts now expose `parser_intermediate_mutation_*`
    aliases so downstream readers do not confuse parser/intermediate evidence
    with full-frame `inflate.sh` evidence.
  - Z3G2 and NSCS02 `SubstrateContract` / trainer / package docs are downgraded
    to `score_improvement_mechanism_status=RESEARCH_ONLY` with
    `runtime_overlay_consumed=false` until full-frame / resizing-chain /
    paired-axis evidence exists.
  - Catalog #291 now defaults to repo-local `.omx/research` review artifacts
    instead of machine-local Claude memory, so strict-flip cannot break clean
    clones or OSS users.

## Explicit non-claims

- No scaffold in this batch is submission-ready.
- No scaffold in this batch has a paired CPU+CUDA exact score.
- No scaffold in this batch should be ranked, retired, or promoted from local
  parser/runtime evidence alone.
- NSCS01/NSCS02/NSCS03 predicted bands are hypotheses for the next exact-eval
  campaign, not score claims.

## Mutable state deliberately not committed

The following tracked-but-mutable files remained unstaged during the
`0b5c55538` push because they are operator/provider state rather than durable
source artifacts:

- `.omx/state/lane_maturity_audit.log`
- `.omx/state/lane_registry.json`
- `experiments/results/_modal_harvest_summary.json`

Preserve their signal through curated `.omx/research/` ledgers before any
cleanup or public release.

## Next gates

- Convert any further reviewer findings into source patches with tests before
  score-ranking these scaffold lanes.
- Run the #291 guard as warn-only until the first full recurrence cycle, then
  strict-flip atomically if the live count remains clean.
- Convert the strongest assumption-violating scaffold into a measured timing
  smoke only after lane claim, recipe review, runtime custody, and explicit
  paired-axis plan.
