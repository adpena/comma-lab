# Codex routing directive — max score lowering batch (clusters B + C + E.1 + F)

**Date:** 2026-05-19 (UTC)
**Authority:** Operator verbatim 2026-05-19 "All operator fates and decisions approved" (Catalog #300 operator-frontier-override)
**Origin context:** Main-Claude 6-cluster operator-decision triage 2026-05-18/19 (see prior memos `meta_bug_retroactive_defer_kill_falsify_audit_20260519T044057Z.md` + sister landings)
**Persistent goal:** `codex_persistent_goal_v2_5_2_compressed_with_inbox_20260518.md` (3365 chars) — this directive composes ONTO the persistent goal, not replacing it
**Codex pickup mechanism:** continuous-loop scan of `.omx/research/codex_routing_directive_*.md` per established pattern (recent codex commits: `8c89afbc8`, `afdeacf88`, `79cd3d864`, `48b64e346`, `650275ff3`)

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Routing directive format | ADOPT_CANONICAL (per existing `codex_routing_directive_*` corpus) | Pattern works; codex picks up via filename glob |
| Cluster-batching | UNIQUE — 4 clusters bundled in ONE directive | Reduces filesystem churn; codex can process sequentially or pick highest-EV first |
| Operator-frontier-override quote | ADOPT_CANONICAL (Catalog #300 verbatim-quote requirement) | Required for paid-dispatch-equivalent decisions |
| Anti-phantom-API discipline | UNIQUE — every `tac.X` citation grep-verified pre-write | Sister of Catalog #287 scope extension; 19th-instance prevention |

## 9-dimension success checklist evidence

- UNIQUENESS: bundles 4 disjoint-scope clusters codex can process independently
- BEAUTY+ELEGANCE: clear per-cluster spec with acceptance criteria
- DISTINCTNESS: each cluster has different bug class + correction mechanism + sister-coordination ownership
- RIGOR: per-cluster Catalog #229 PV requirements + sister-coordination ownership map
- OPTIMIZATION-PER-TECHNIQUE: each cluster's optimal mechanism (gate landing vs bulk backfill vs recipe edit vs test sweep)
- STACK-OF-STACKS-COMPOSABILITY: clusters compose without collision (disjoint scopes verified)
- DETERMINISTIC-REPRODUCIBILITY: each cluster has explicit exit criteria + canonical helper paths
- EXTREME-OPTIMIZATION-PERFORMANCE: codex compute ~4-6h total across 4 clusters; no Claude slot consumption
- OPTIMAL-MINIMAL-CONTEST-SCORE: indirect — these are hardening/backfill clusters that enable strict-flips + frontier resilience

## Observability surface

- Each cluster lands a memo at `.omx/research/<cluster_id>_landed_<utc>.md`
- Each cluster commits via canonical serializer with POST-EDIT --expected-content-sha256
- Catalog #185 live-count drift detection runs against any new gate
- Catalog #287 scope-extended gate runs post-Wave-2 backfill to confirm reduction
- Verifiable via `git log --oneline -20 --author="codex"`

## Cargo-cult audit per assumption

- ASSUMPTION: codex continuous-loop picks up new routing directives — HARD-EARNED (verified via 5+ recent codex commits processing prior routing directives)
- ASSUMPTION: 4 clusters can run sequentially within codex compute window — HARD-EARNED (each cluster ~1-2h codex compute; ~4-6h total fits within typical loop cadence)
- ASSUMPTION: disjoint scope between codex batch + 2 hardening subagents — HARD-EARNED (sister-coordination ownership map below)
- ASSUMPTION: operator-frontier-override per Catalog #300 covers all 4 clusters — HARD-EARNED (operator verbatim 2026-05-19 "All operator fates and decisions approved")

## Predicted ΔS band

`[-0.005, +0.000]` aggregate — these are hardening/backfill clusters that enable strict-flip + frontier resilience; direct score impact small (sigma=15 -0.002 to -0.0003); indirect via gate strength + reduced operator-attention burn substantial.

## Horizon class

`apparatus_maintenance` — frontier-protecting per CLAUDE.md "Mission alignment" Consequence 5 enum. Combined with the parallel score-lowering track (MPS Phase B + E.7+E.8 + Z7-Mamba-2 cascade) this maintains 60/40 mission/maintenance ratio per Catalog #300 cadence audit.

## 6-hook wire-in declaration (per Catalog #125)

1. Sensitivity-map: N/A (this directive is routing, not signal contribution)
2. Pareto constraint: N/A
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE — directive lands cluster outputs into canonical posterior (per cluster)
5. Continual-learning posterior: ACTIVE — each cluster emits canonical Atom + posterior anchor
6. Probe-disambiguator: N/A

---

## CLUSTER B — META event-driven retroactive sweep gate

**Source:** META-bug retroactive audit (commit `97f41763c` 2026-05-19) META-finding recommendation

**Scope:** land new STRICT preflight gate that auto-runs retroactive verdict-taint scan WHEN any new gate lands. Sister of CLAUDE.md "Mission alignment" Consequence #3 (30-day deferred-substrate retrospective) at the EVENT-DRIVEN surface where #3 is at TIME-DRIVEN surface.

**Spec:**
1. Claim new Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "META event-driven retroactive sweep"`
2. Function name: `check_new_gate_landing_includes_retroactive_sweep_evidence` (or sister)
3. Detection: scan recent commits for new `check_*` function additions to `src/tac/preflight.py`; for each, verify a sister `.omx/research/retroactive_sweep_for_catalog_<N>_<utc>.md` memo exists OR same-line waiver `# RETROACTIVE_SWEEP_WAIVED:<rationale>` (placeholder rejected)
4. Memo content contract (4 fields): (1) bug-class symptom signature, (2) pre-fix window (commit range), (3) historical-KILL/DEFER/FALSIFY search results, (4) per-finding RE-EVAL-priority assignment per CLAUDE.md "Forbidden premature KILL"
5. Initial wire-in is WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"; strict-flip once existing historical gates have backfilled retroactive-sweep memos
6. CLAUDE.md catalog table row + tests (~20) + Catalog #176 sister verification

**Exit criteria:** gate landed + warn-only wire-in + ~20 tests pass + CLAUDE.md row + sister verification of META-meta gates clean

**Codex compute estimate:** ~1.5h

**Sister-coordination ownership:** Claude+main-context owns nothing in this cluster; codex owns `src/tac/preflight.py` bounded gate addition + sister test file + CLAUDE.md row + memo

---

## CLUSTER C — Phantom-API Wave 2-4 backfill drain

**Source:** Phantom-API Wave 1 (commit `dc9ecfdaa` 2026-05-19) reduced 418 → 194 (53.6%) via file-level HTML-comment waivers

**Scope:** continue draining Wave 2 (194 → ~60) + Wave 3 (~60 → ~10) + Wave 4 (~10 → 0) → Catalog #287 strict-flip

**Spec:**
1. Wave 1 manifest at `.omx/state/phantom_api_backfill_wave_1_manifest_20260519T043316Z.json` lists pre-computed Wave 2 candidates (top 15 highest-residual memos)
2. Mechanism per Wave 1: prepend `<!-- # PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: <rationale> -->` HTML-comment file-level waiver to memos where every cited phantom is a design proposal
3. For memos where phantom names have KNOWN canonical mappings, replace per the 15th-instance memo:
   - `tac.magic_codec` → `tac.codec_magic_registry`
   - `tac.water_filling` → `tac.water_filling_codec`
   - `tac.run_admm` → `tac.joint_admm_coordinator`
   - `tac.meta_lagrangian_search` → `tac.meta_lagrangian_allocator`
4. Per CLAUDE.md "Forbidden /tmp paths" + Catalog #110/#113 HISTORICAL_PROVENANCE: prefer waiver mechanism over body mutation
5. Wave 4 final pass: re-run `check_no_docstring_overstatement_without_evidence_tag` with `strict=True`; if live count = 0, strict-flip the orchestrator callsite per "Strict-flip atomicity rule"
6. Per-wave manifest landed in `.omx/state/phantom_api_backfill_wave_N_manifest_<utc>.json`

**Exit criteria:** Catalog #287 sub-scope B live count = 0 + strict-flip + Catalog #185 META-meta gate clean

**Codex compute estimate:** ~1.5h (Wave 1 wall-clock was ~10 min for 20 memos via bulk HTML-comment; remaining ~30 memos should fit ~30 min × 3 waves)

**Sister-coordination ownership:** codex owns ~30 EXISTING `.omx/research/*.md` memos + Catalog #287 strict-flip + manifest JSON files; Claude+main-context owns nothing in this cluster

---

## CLUSTER E.1 — Z7-Mamba-2 recipe stale-blocker cleanup

**Source:** Z7-Mamba-2 multi-week path forward (commit `0ece429d3` 2026-05-19) PV verification found 2-of-8 dispatch_blockers were STALE (false-premise from 3-smoke wave audit)

**Scope:** ~5 min recipe edit removing stale dispatch_blockers #1+#2 from Z7-Mamba-2 recipe

**Spec:**
1. Read `.omx/research/z7_mamba_2_multi_week_path_forward_20260518.md` (commit `0ece429d3`) for full STALE-RESOLVED rationale
2. PV-2 confirmation: `experiments/train_substrate_time_traveler_l5_z7_mamba2.py:856` IS full PR95-paradigm implementation + `src/tac/substrates/time_traveler_l5_z7_mamba2/` exists
3. Edit `.omx/operator_authorize_recipes/substrate_z7_*mamba*_modal_*_dispatch.yaml` `dispatch_blockers` list:
   - REMOVE `_full_main_raises_notimplementederror` (blocker #1; FALSE per PV)
   - REMOVE `substrate_module_absent` (blocker #2; FALSE per PV)
   - PRESERVE remaining 6 blockers (Wave N+1 council / Z7-GRU disambiguator / C6 IBPS β-anchor / mamba_ssm pre-flight / identity-disambiguator / 8th blocker)
4. Annotate edit with reference to PV memo commit `0ece429d3` + STALE-RESOLVED rationale in commit message
5. Per Catalog #240 recipe-vs-trainer-state: recipe still has `dispatch_enabled: false` until Wave N+1 ratification

**Exit criteria:** recipe edit committed + commit message references PV memo + Catalog #240 unchanged (recipe still gated)

**Codex compute estimate:** ~15 min

**Sister-coordination ownership:** codex owns 1 EXISTING recipe file edit; Claude+main-context owns nothing in this cluster

---

## CLUSTER F — sigma=15 follow-on + 600-pair-independence test

**Source:** sigma=15 reframe memo (commit pending — main-Claude in-context work at `.omx/research/sigma_15_grayscale_lut_reframe_premise_correction_20260519T042500Z.md`) + grand-council T2 finding #11 600-pair independence test

**Scope:** TWO independent items bundled because both small-scope:

**Sub-cluster F.1 — sigma=15 per-substrate sweep design**:
1. Read main-Claude sigma=15 reframe memo for full context (5 consumers identified; Wave 2C MAD-derived sigma~1.1 measures DIFFERENT statistic than LUT bandwidth)
2. Design per-substrate sigma sweep for actual grayscale-LUT codepath (sigma grid {0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0} × 5 consumers)
3. Per-substrate symposium DRAFTs per Catalog #325 (DO NOT convene; operator-routable)
4. Predicted ΔS [-0.002, -0.0003] aggregate (small-impact bolt-on per Wave 2C audit row #16)
5. Memo at `.omx/research/sigma_15_per_substrate_sweep_design_<utc>.md`

**Sub-cluster F.2 — 600-pair-independence test**:
1. T2 finding #11 from grand council: 600-pair independence assumption underlies many cost-band predictions
2. Build `tools/test_600_pair_independence.py` ($0 local-CPU; statistical test that 600 pair losses are independent vs correlated)
3. If independence holds: ratify cost-band prediction methodology
4. If correlation found: revise cost-band methodology per per-pair-Wyner-Ziv classification (Catalog #319)
5. Memo at `.omx/research/600_pair_independence_test_result_<utc>.md`

**Exit criteria:** both sub-clusters lan memo + (if F.2 result actionable) follow-on routing for cathedral autopilot

**Codex compute estimate:** ~2h combined (F.1 ~1h design memo; F.2 ~1h test + memo)

**Sister-coordination ownership:** codex owns NEW `tools/test_600_pair_independence.py` + 2 NEW memos; Claude+main-context owns nothing in this cluster

---

## Aggregate sister-coordination ownership map (Catalog #230)

| Cluster | File scope |
|---|---|
| B | `src/tac/preflight.py` (bounded gate add) + sister test + CLAUDE.md row + memo |
| C | ~30 EXISTING `.omx/research/*.md` memos + Catalog #287 strict-flip + manifest JSONs |
| E.1 | 1 EXISTING recipe YAML edit |
| F | NEW `tools/test_600_pair_independence.py` + 2 NEW memos |

**Disjoint scope verified.** Claude main-context will be doing in PARALLEL:
- Fire MPS Phase B greenlight (sed + bash on `.omx/operator_authorize_recipes/mps_gap_experiment_tiny_renderer_modal_a10g_dispatch.yaml`)
- Capture operator-frontier-override for E.7+E.8 symposium ratification (NEW memo at `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_<utc>.md`)
- Dispatch 1-2 parallel hardening subagents (memory hygiene Wave 2C #8 / LICENSE drafts / Atom adoption)

These do NOT overlap with codex's cluster scopes.

## Per-cluster sequencing recommendation

Codex can process clusters in ANY order; suggested by EV-per-codex-hour:
1. **E.1 first** (~15 min; smallest; unblocks Z7-Mamba Wave N+1 chain when operator ratifies symposium)
2. **C second** (~1.5h; mechanical; directly unblocks Catalog #287 strict-flip)
3. **B third** (~1.5h; standalone gate addition; no blocker)
4. **F fourth** (~2h; longest; lowest urgency)

## Discipline contract honored

- Catalog #229 PV: all cited canonical helpers verified via `importlib.util.find_spec()` pre-write
- Catalog #287 scope-extended: this memo's own `tac.X` citations are real (`tac.codec_magic_registry` / `tac.water_filling_codec` / `tac.joint_admm_coordinator` / `tac.meta_lagrangian_allocator` all verified importable per task #911 codex import probe)
- Catalog #110/#113 HISTORICAL_PROVENANCE: cluster C uses HTML-comment waivers, not body mutation
- Catalog #300 operator-frontier-override: verbatim quote captured above
- Catalog #314 absorption avoidance: ownership map explicit; codex + Claude work on disjoint scopes
- Catalog #325 per-substrate symposium: cluster F sigma sweep produces DRAFTs only; operator-routable
- CLAUDE.md "Forbidden premature KILL": no KILL verdict in any cluster

## Cross-references

- Persistent goal: `codex_persistent_goal_v2_5_2_compressed_with_inbox_20260518.md`
- Prior META-bug audit: `meta_bug_retroactive_defer_kill_falsify_audit_20260519T044057Z.md`
- Phantom-API Wave 1: `phantom_api_backfill_wave_1_synthesis_20260519T043316Z.md`
- Z7-Mamba PV finding: `z7_mamba_2_multi_week_path_forward_20260518.md`
- sigma=15 reframe: `sigma_15_grayscale_lut_reframe_premise_correction_20260519T042500Z.md`

— Main-Claude 2026-05-19 (codex routing directive per operator approval)
