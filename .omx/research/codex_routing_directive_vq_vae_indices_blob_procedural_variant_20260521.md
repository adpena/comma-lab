# Codex Routing Directive — VQ-VAE indices_blob procedural-codebook variant extension (L0 scaffold; DESIGN + BUILD; no paid dispatch)

**Date**: 2026-05-21T04:52:00Z (UTC)
**Authority**: Operator blanket approval 2026-05-20 + WAVE-3-REVERSE-CODEX-ROUTING-DIRECTIVES-FANOUT fan-out per CODEX CROSS-POLLINATION audit `aafac7c84` §15.4 REVERSE-DIRECTIVE #4
**For consumption by**: codex CLI subagent (Pattern A detached BG invocation)
**Source draft**: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 REVERSE-DIRECTIVE #4
**Lane**: `lane_codex_vq_vae_indices_blob_procedural_variant_20260521`

## Operator directive

Extend the VQ-VAE substrate with a procedural-codebook variant targeting the **192-byte indices_blob** (RAW int16 codebook indices) identified by PARSER-SAFE EXTENSION (main-thread commit `d0bf3ce37`) as parser-safe + score-affecting. This is the **rank #4** candidate in `procedural_replacement_surface_matrix_landed_20260521_codex.md`.

The motivation: canonical equation #26 `procedural_codebook_from_seed_compression_savings_v1` predicts REPLACEMENT savings `ΔS = -25 * (N_codebook - K_seed) / 37_545_489` for in-domain contexts (per commit `79f1ba387` domain refinement event 8); the 192-byte indices_blob is parser-safe and score-affecting per the 4-substrate static classification at commit `d0bf3ce37`. A procedural-codebook variant (derive indices from a small seed via deterministic procedure) MAY apply equation #26's prediction, OR may instead fall under the new equation `procedural_predictor_plus_residual_correction_savings_v1` (slot 1 commit `af36cd72` + ratification `098d8a31c`) IF a residual correction layer is required.

The variant is DESIGN + L0 scaffold ONLY; NO paid GPU dispatch per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable. The Catalog #325 per-substrate symposium MUST occur BEFORE paid dispatch (separate operator-approval gate).

## Pre-flight (Catalog #229 premise verification)

Read these files in full BEFORE any design verdict:

- `src/tac/substrates/vq_vae/` (canonical implementation; full package)
- `experiments/train_substrate_vq_vae.py` (canonical trainer)
- `.omx/research/parser_safe_methodology_extension_landed_20260520.md`
- `.omx/research/parser_safe_subset_smoke_landed_20260520.md`
- `.omx/research/procedural_replacement_surface_matrix_landed_20260521_codex.md`
- Sister DP1 procedural-variant landings:
  - `.omx/research/dp1_paired_smoke_recipes_landed_20260521_codex.md`
  - `.omx/research/dp1_procedural_paired_harvest_planner_landed_20260521_codex.md`
- Canonical equation #26 INCLUDED contexts: `src/tac/canonical_equations/procedural_codebook_savings.py` + `src/tac/canonical_equations/procedural_codebook_savings_domain_refinement.py` + `.omx/state/canonical_equations_registry.jsonl` (filter to `equation_id=procedural_codebook_from_seed_compression_savings_v1`)
- NEW equation `procedural_predictor_plus_residual_correction_savings_v1` design memo (commit `d3e63bbe9`) + ratification memo (commit `098d8a31c`)
- Catalog #344 strict gate behavior on new domain_refinement events
- Catalog #318 raw-byte-authority guard (consumers must respect chain-rule discipline)
- Probe outcomes: `tools/check_predecessor_probe_outcome.py --substrate vq_vae` (Catalog #313)
- DP1 streamer bug investigation memo `.omx/research/ea42b3102_dp1_harvest_bug_3_candidate_fix_matrix.md` (filename may differ; commit `ea42b3102`) for cross-cascade lessons

## Deliverables

1. **Design memo** at `.omx/research/vq_vae_procedural_codebook_variant_design_<UTC>.md` with the canonical substrate design memo contract enforced by:
   - Catalog #290 `## Canonical-vs-unique decision per layer` section
   - Catalog #294 `## 9-dimension success checklist evidence` section
   - Catalog #303 `## Cargo-cult audit per assumption` section
   - Catalog #305 `## Observability surface` section
   - Catalog #296 Dykstra-feasibility check on `## Predicted ΔS band` section
   - Catalog #309 `horizon-class:` declaration (per matrix rank #4 — likely `parser_pursuit` if a new class is defined; otherwise `plateau_adjacent` or `frontier_pursuit` per CLAUDE.md "HORIZON-CLASS evaluation axis")
2. **Paradigm classification** per CODEX CROSS-POLLINATION audit §15.1 Insight 1 (3-paradigm taxonomy):
   - **REPLACEMENT-UPSTREAM** (equation #26 applies; codebook bytes REMOVED and seed bytes ADDED)
   - **RESIDUAL-CORRECTION-DOWNSTREAM** (new equation applies; predictor + residual correction stacked)
   - **REMOVAL** (refuse: score-affecting per parser-safe matrix, so removal alone would degrade score)
3. **Catalog #325 per-substrate symposium memo** at `.omx/research/council_grand_council_vq_vae_procedural_codebook_variant_<YYYYMMDD>.md` OR explicit `research_only=true` opt-out tag in the design memo declaring symposium DEFERRED
4. **Trainer extension** in `experiments/train_substrate_vq_vae.py`:
   - `VQ_PROCEDURAL_CODEBOOK_REPLACEMENT` env knob (default off; opt-in for variant)
   - `--enable-procedural-codebook` argparse flag
   - `--procedural-seed-bytes <int>` argparse flag (default per design memo)
   - Tier 1 per CLAUDE.md "Production-hardened dispatch optimization protocol": `--enable-autocast-fp16` / TF32 / `--enable-torch-compile` / no_grad eval already declared (verify; do NOT regress)
   - Canonical scorer-loss helper routing per Catalog #164
5. **Procedural-aware inflate vendor** following the DP1 pattern (vendor the procedural codebook generator into `submission_dir/src/tac/...` so inflate is self-contained per Catalog #295 empty-PYTHONPATH test)
6. **Operator-gated paired-smoke recipes** at:
   - `.omx/operator_authorize_recipes/substrate_vq_vae_procedural_codebook_modal_<gpu>_dispatch.yaml`
   - `.omx/operator_authorize_recipes/substrate_vq_vae_procedural_codebook_paired_baseline_modal_<gpu>_dispatch.yaml`
   - BOTH with `dispatch_enabled: false` initially + `research_only: true` until Catalog #325 symposium PROCEED-unconditional verdict lands
   - Recipe schema MUST satisfy Catalog #170/#171/#172/#181/#182/#215/#244 sister gates
7. **Tests** (≥10 tests covering):
   - Happy path: procedural codebook reconstructs original indices when seed is correct
   - Edge cases: seed-size 0 / max / out-of-range
   - Round-trip determinism: same seed always produces same indices
   - Catalog #318 raw-byte-authority guard regression
   - Catalog #287 evidence-tag in any test docstring claiming performance
   - Trainer smoke (synthetic): `_smoke_main` produces non-crashing output
   - Archive grammar: procedural variant archive parses cleanly via `tac.packet_compiler.deterministic_compiler` (sister Catalog #158)
   - Inflate-time consumer: vendored procedural generator deterministically reproduces indices
   - Catalog #220 byte-mutation: mutating the seed changes the rendered output (operational mechanism proof)
   - Catalog #272 distinguishing-feature integration contract: declare + verify
8. **Probe-outcomes ledger** register PROCEED verdict at `.omx/state/probe_outcomes.jsonl` AFTER design + L0 scaffold land (Catalog #313 sister discipline)

## Discipline (per CLAUDE.md non-negotiables)

- Catalog #229 PV (read full canonical implementation + canonical equation surfaces + sister DP1 cascade pre-design)
- Catalog #287 evidence-tag for every empirical / predicted claim (`[empirical:<artifact>]` / `[predicted]` / `[advisory only]`)
- Catalog #318 raw-byte-authority guard (consumers MUST emit `CandidateModificationSpec` + `grammar_aware_operator` typed responses; no `byte_modifications: Mapping[int, float]` raw byte APIs)
- Catalog #220 substrate L1+ scaffold operational mechanism (variant MUST declare `score_improvement_mechanism_status=OPERATIONAL` AND inflate-time consumer MUST modify rendered frames)
- Catalog #325 per-substrate symposium MANDATORY before paid dispatch
- Catalog #344 canonical equation registry: design memo MUST cite equation #26 OR new equation `procedural_predictor_plus_residual_correction_savings_v1` explicitly
- Catalog #359 misapplication-to-residual-hybrid guard: if RESIDUAL-CORRECTION paradigm chosen, design memo MUST route through the NEW equation, NOT equation #26
- Catalog #117/#157/#174 canonical commit serializer with POST-EDIT `--expected-content-sha256`
- Catalog #206 checkpoint every ~5 tool uses; final `--step complete --status complete` checkpoint
- Catalog #119 Co-Authored-By trailer (codex's commits are internal subagent commits)
- Catalog #234 substantive commit message bodies
- Catalog #248 zero residual conflict markers
- Catalog #340 sister-checkpoint guard PROCEED required at every commit
- 6-hook wire-in declaration per Catalog #125

## 6-hook wire-in declaration per Catalog #125 (for the variant's runtime artifact, not this directive memo)

- Hook #1 sensitivity-map: **ACTIVE if RESIDUAL-CORRECTION paradigm** (per-byte sensitivity of residual stream); **N/A if pure REPLACEMENT**
- Hook #2 Pareto constraint: **ACTIVE** (procedural seed bytes vs codebook indices bytes is a rate-vs-distortion tradeoff)
- Hook #3 bit-allocator: **ACTIVE** (`predicted_archive_bytes_delta` from canonical equation #26 / new equation)
- Hook #4 cathedral autopilot dispatch: **ACTIVE** (variant participates in autopilot ranking via the cathedral consumer `canonical_equation_lookup_consumer`)
- Hook #5 continual-learning posterior: **ACTIVE** (empirical anchors from variant smoke dispatches update equation #26 / new equation posterior via `tac.canonical_equations.update_equation_with_empirical_anchor`)
- Hook #6 probe-disambiguator: **ACTIVE** (paired baseline + procedural recipes empirically disambiguate REPLACEMENT vs RESIDUAL-CORRECTION paradigm)

## Scope limits

DO NOT:
- Fire paid GPU dispatch (DESIGN + L0 scaffold scope; Catalog #325 symposium gate)
- Mutate canonical equation #26 mathematical predicate or `_INCLUDED_CONTEXTS` set without explicit operator authorization (preserved per Catalog #110/#113 APPEND-ONLY; new contexts via NEW `domain_refinement` event ONLY)
- Mutate sister DP1 recipes / trainer / lane driver per Catalog #110/#113
- Spawn nested subagents
- Modify CLAUDE.md (canonical contract surface)
- Push to origin (codex commits land via canonical serializer; operator pushes)
- Add file-level waivers on the gate it itself triggers (placeholder-rationale rejection per Catalog #287)
- Mutate other codex memos per Catalog #110/#113 APPEND-ONLY

## Estimated cost

- $0 GPU (DESIGN + L0 scaffold; NO paid dispatch)
- ~3h wall-clock

## Cross-references

- CODEX CROSS-POLLINATION audit memo: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 REVERSE-DIRECTIVE #4
- PARSER-SAFE EXTENSION landing: `.omx/research/parser_safe_methodology_extension_landed_20260520.md` (commit `d0bf3ce37`)
- Procedural replacement surface matrix: `.omx/research/procedural_replacement_surface_matrix_landed_20260521_codex.md`
- Equation #26 domain refinement event 8: commit `79f1ba387`
- New equation slot 1 commit: `af36cd72` + ratification `098d8a31c` + design memo `d3e63bbe9`
- Sister DP1 procedural variant: commits `b93c15afd` / `940a77e2f` / `9aab2a177`
- Aggregate WAVE-3-FAN-OUT landing: `.omx/research/reverse_codex_routing_directives_fan_out_landed_20260521.md`

## Operator-routable instruction

```bash
codex /goal --skill codex-cli-runtime \
    --input .omx/research/codex_routing_directive_vq_vae_indices_blob_procedural_variant_20260521.md \
    --goal "VQ-VAE indices_blob procedural-codebook variant DESIGN + L0 scaffold per WAVE-3 REVERSE-DIRECTIVE #4"
```

OR Pattern A detached invocation per CLAUDE.md "Codex CLI invocation":

```bash
mkdir -p .omx/tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o .omx/tmp/codex_runs/vq_vae_procedural_codebook_variant.last.txt \
    "$(cat .omx/research/codex_routing_directive_vq_vae_indices_blob_procedural_variant_20260521.md)" \
    2>&1 | tee .omx/tmp/codex_runs/vq_vae_procedural_codebook_variant.log > /dev/null
' < /dev/null > .omx/tmp/codex_runs/vq_vae_procedural_codebook_variant.outer.log 2>&1 &
disown
```
