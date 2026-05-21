# Codex Routing Directive — DP1 paired-smoke recipes parity audit vs procedural variant

**Date**: 2026-05-21T04:50:00Z (UTC)
**Authority**: Operator blanket approval 2026-05-20 ("all operator decisions and approval granted and provided fully and completely") + WAVE-3-REVERSE-CODEX-ROUTING-DIRECTIVES-FANOUT fan-out per CODEX CROSS-POLLINATION audit `aafac7c84` §15.4 REVERSE-DIRECTIVE #2
**For consumption by**: codex CLI subagent (Pattern A detached BG invocation OR Pattern B Agent wrapper per CLAUDE.md "Codex CLI invocation")
**Source draft**: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 REVERSE-DIRECTIVE #2
**Lane**: `lane_codex_dp1_paired_smoke_parity_audit_20260521`

## Operator directive

Audit the DP1 paired-smoke recipe set for parity with the procedural-variant landing at commit `b93c15afd` (Wire DP1 procedural paired-smoke recipes) and the cache-source re-routing at commit `940a77e2f` (Route DP1 paired smoke through cache source) and the recipe activation at commit `9aab2a177` (Activate DP1 paired-smoke recipes).

The motivation: a procedural variant was wired and activated through 3 sibling commits. The 3 sibling recipes (baseline + procedural + ablation) must remain at parity on EVERY axis EXCEPT the deliberate procedural-vs-baseline distinguishing knob. Recent sister commits `e9ec227bd` (Gate DP1 paired harvest on training call status) and `0f7ea70a8` (Verify DP1 paired sentinel equivalence) further amended the recipe surface; this audit verifies they did not introduce drift between sisters.

This audit is a no-paid-dispatch reconnaissance pass with the canonical 6-step contract per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable.

## Pre-flight (Catalog #229 premise verification)

Read these files in full BEFORE any audit verdict:

- `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_*.yaml` (all 3+ DP1 recipes; baseline + procedural + any ablation/paired sisters)
- `scripts/remote_lane_substrate_pretrained_driving_prior.sh` (lane driver)
- `experiments/train_substrate_pretrained_driving_prior.py` (canonical trainer)
- Sister codex landings:
  - `dp1_paired_smoke_recipes_landed_20260521_codex.md`
  - `dp1_procedural_paired_harvest_planner_landed_20260521_codex.md`
  - `dp1_streamer_no_chunk_ids_dispatch_failure_20260521T031333Z_codex.md`
  - `dp1_cache_source_paired_dispatch_landed_*_codex.md` (if exists; sister of `2c326a2d6` + `d4e5abd0b` Modal call recording commits)
- Sister Modal call ledger rows: `.omx/state/modal_call_id_ledger.jsonl` filter to `lane_id` starting with `lane_dp1_` for the last 7 days
- Probe outcomes: `tools/check_predecessor_probe_outcome.py --substrate pretrained_driving_prior` (Catalog #313)

## Deliverables

1. **Audit memo** at `.omx/research/dp1_paired_smoke_recipes_parity_audit_<UTC>_codex.md`
2. **Per-recipe verdict** in tabular form: `(recipe_name, status, drift_class)` where `status ∈ {PARITY, DRIFT_DETECTED, RESOLVED_VIA_SISTER_COMMIT, INTENTIONAL_DIFFERENCE}` and `drift_class` enumerates the canonical axes (`gpu` / `min_vram_gb` / `pyav_decode_strategy` / `target_modes` / `canary_status` / `env_overrides` / `lane_script` / `cost_band` / `predicted_band` / `predicted_band_validation_status` / `dispatch_enabled` / `research_only` / `smoke_only`)
3. **For each DRIFT_DETECTED**: surface root cause + propose 1-3 LOC fix recipe-side (do NOT mutate trainer)
4. **Post-fix verification** (apply fixes to local working tree; do NOT commit):
   - `tools/local_pre_deploy_check.py --strict` passes for all 3+ recipes
   - `tools/operator_authorize.py --dry-run --recipe <each>` exits rc=0 (or documents structural rc≠0 reasons unrelated to drift)
5. **Catalog #357 Tier B compliance check**: recipes are operator-authorize tier, NOT cathedral consumer tier B; skip if N/A but DOCUMENT the verdict
6. **Catalog #325 per-substrate symposium check**: verify DP1 symposium memo exists at `.omx/research/council_*_pretrained_driving_prior_*_<YYYYMMDD>.md` within last 14 days OR document the gap (operator-routable next-action)
7. **Catalog #324 predicted_band_validation_status audit**: every recipe with `predicted_band` field MUST carry `validated_post_training` or `pending_post_training` per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density"
8. **Catalog #244 NVML env block check**: lane driver carries canonical 3-export block per CLAUDE.md "Production-hardened dispatch optimization protocol" Tier 2
9. **Catalog #240 recipe-vs-trainer-state consistency check**: trainer's `_full_main` is implemented (not `raise NotImplementedError`) AND recipes accurately reflect dispatchability

## Discipline (per CLAUDE.md non-negotiables)

- Catalog #229 PV (read full recipe state pre-audit)
- Catalog #110/#113 APPEND-ONLY (audit memo only; do not mutate landed recipes; landed Modal calls in ledger are HISTORICAL_PROVENANCE)
- Catalog #287 evidence-tag for every observation (`[empirical:<artifact>]` / `[advisory only]` / `[predicted]`)
- Catalog #270 dispatch optimization protocol scope clarification: DP1 is substrate trainer, NOT tool dispatch — Tier 1/2/3 fully applicable
- Catalog #117/#157/#174 canonical commit serializer with POST-EDIT `--expected-content-sha256` for any commit
- Catalog #206 checkpoint every ~5 tool uses; final `--step complete --status complete` checkpoint
- Catalog #248 zero residual conflict markers in audit memo
- Catalog #340 sister-checkpoint guard PROCEED required at write time
- 6-hook wire-in declaration per Catalog #125

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: **N/A** (audit memo, no signal contribution)
- Hook #2 Pareto constraint: **N/A**
- Hook #3 bit-allocator: **N/A**
- Hook #4 cathedral autopilot dispatch: **ACTIVE** (DRIFT_DETECTED verdicts surface for operator review; PARITY verdicts ratify the existing 3-sibling recipe set so autopilot can rank them apples-to-apples)
- Hook #5 continual-learning posterior: **ACTIVE** (audit verdict appended to probe_outcomes_ledger per Catalog #313 if DRIFT_DETECTED found)
- Hook #6 probe-disambiguator: **N/A**

## Scope limits

DO NOT:
- Fire paid GPU dispatch (audit is observability-only)
- Mutate landed recipes (work in scratch clone or `/tmp/` work tree per CLAUDE.md "Forbidden /tmp paths" — `/tmp/` is acceptable for ephemeral local scratch per CLAUDE.md exception)
- Mutate Modal call ledger or active dispatch claims (those are HISTORICAL_PROVENANCE)
- Push to origin
- Re-run smoke dispatches (the goal is recipe surface drift, not score validation)
- Mutate sister memos per Catalog #110/#113 APPEND-ONLY

## Estimated cost

- $0 GPU
- ~1h wall-clock

## Cross-references

- CODEX CROSS-POLLINATION audit memo: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 REVERSE-DIRECTIVE #2
- Sister codex DP1 cascade landings (read in pre-flight)
- Aggregate WAVE-3-FAN-OUT landing: `.omx/research/reverse_codex_routing_directives_fan_out_landed_20260521.md`

## Operator-routable instruction

```bash
codex /goal --skill codex-cli-runtime \
    --input .omx/research/codex_routing_directive_dp1_paired_smoke_parity_audit_20260521.md \
    --goal "Audit sister DP1 paired-smoke recipes for parity per WAVE-3 REVERSE-DIRECTIVE #2"
```

OR Pattern A detached invocation per CLAUDE.md "Codex CLI invocation — NON-NEGOTIABLE":

```bash
mkdir -p .omx/tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o .omx/tmp/codex_runs/dp1_paired_smoke_parity_audit.last.txt \
    "$(cat .omx/research/codex_routing_directive_dp1_paired_smoke_parity_audit_20260521.md)" \
    2>&1 | tee .omx/tmp/codex_runs/dp1_paired_smoke_parity_audit.log > /dev/null
' < /dev/null > .omx/tmp/codex_runs/dp1_paired_smoke_parity_audit.outer.log 2>&1 &
disown
```
