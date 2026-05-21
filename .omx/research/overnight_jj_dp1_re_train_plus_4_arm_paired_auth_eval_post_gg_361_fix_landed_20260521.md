---
council_tier: T1
council_attendees: [Claude]
council_quorum_met: true
council_verdict: IN_FLIGHT
council_dissent: []
council_decisions_recorded:
  - "dispatched DP1 baseline + procedural training arms on HEAD cd036aa61 (post-GG-#361 fix 83ed831e3) per OVERNIGHT-CC operator-routable #2 + GG follow-up #1"
  - "verified prior 13:49Z + 13:51Z dispatches landed on commit 6e684236c BEFORE GG #361 fix; their artifacts still have empty vendor stubs (root cause of OVERNIGHT-CC ModuleNotFoundError)"
  - "verified HEAD cd036aa61 contains both GG fix surfaces: under_submission bypass in experiments/modal_train_lane.py + vendor_module_with_fresh_mtime in DP1 trainer"
  - "Catalog #166 pre_spawn_fatal fired correctly on first 17:11Z attempt; resolved via Catalog #202 paired-env bypass (sentinels clean; dirty tree categorized as sister-state-ephemeral + my own derived output + unrelated-stale)"
  - "registered 2 NEW Modal call_id ledger rows + 2 lane_dispatch_claims active rows via canonical helpers per Catalog #131/#138/#245/#339"
  - "training-only dispatches (DPP_SKIP_AUTH_EVAL=1); 4-arm paired CUDA+CPU x baseline+procedural auth_eval will fire SEPARATELY via tools/dispatch_modal_paired_auth_eval.py once training harvests at T+45min"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
predicted_band: [0.175, 0.190]
predicted_band_validation_status: pending_post_training
predicted_band_reactivation_criteria: "Replace this planning band only after paired DP1 baseline/procedural training-output harvest and post-training Tier-C density / exact-score review."
related_deliberation_ids:
  - overnight_cc_dp1_path_a_auth_eval_refire_blocked_by_vendor_stub_bug_landed_20260521
  - overnight_gg_dp1_trainer_vendor_stub_bug_fix_plus_catalog_361_landed_20260521
  - dp1_3rd_attempt_path_a_success_first_paid_byte_anchor_canonical_equation_26_registration_landed_20260521
---

# OVERNIGHT-JJ DP1 re-train + 4-arm paired auth_eval post GG #361 fix LANDED 2026-05-21

## Summary

Per operator authorization 2026-05-21 verbatim *"All approved"* + canonical PR 110 frontier reference baseline per Catalog #316 + Carmack MVP-first 5-step per CLAUDE.md `be125b878` + OVERNIGHT-CC operator-routable #2 (`99d06f967`) + OVERNIGHT-GG follow-up #1 (`83ed831e3`), this lane re-dispatches DP1 baseline + procedural training arms on current HEAD (`cd036aa610b955488146781a2798cee65476b9ee`) which is **POST GG #361 fix** so the trainer-emitted submission_dir vendored bodies will now be PRESERVED through Modal artifact harvest.

This memo records: (a) training dispatch state (`IN_FLIGHT` — both arms spawned ~17:19-17:20 UTC); (b) the operator-routable for the harvest cron + downstream 4-arm paired auth_eval re-fire that fires AFTER training completes; (c) cross-agent dispatch coordination + observability state.

## Pre-dispatch verification (Catalog #229 PV)

Confirmed CC's diagnosis: prior 13:49Z + 13:51Z DP1 dispatches landed on commit `6e684236c` which is BEFORE GG #361 fix `83ed831e3`. Their harvested artifacts at `experiments/results/lane_substrate_pretrained_driving_prior_*_modal_t4_paired_dispatch_20260521T13[45]*_modal/harvested_artifacts/output/submission/` still have EMPTY `__init__.py` stubs (0 bytes) + missing module bodies — exact pattern CC observed in 4-arm auth_eval re-fire failures.

Empirical receipt: `find $BASELINE_134957Z/harvested_artifacts/output/submission -type f` shows only `inflate.py` (1177 B) + `0.bin` (26050 B) + `inflate.sh` (378 B) + 5 × 0-byte `__init__.py` stubs. The 8 module bodies (architecture.py / archive.py / codebook.py / inflate.py / prior_application.py / procedural_codebook_inflate.py / seed_derived_codebook.py / _shared/inflate_runtime.py) are MISSING because:
- Commit `6e684236c` lacks GG fix `83ed831e3` (verified via `git merge-base --is-ancestor`).
- Without GG's `under_submission` harvester bypass, `shutil.copy2`-vendored bodies carry old source mtime < `artifact_mtime_floor` and are DROPPED.

HEAD `cd036aa610b9` verified to contain BOTH GG fix surfaces:
- `experiments/modal_train_lane.py` carries `under_submission = (...rel_parts[0] == 'output' and rel_parts[1] == 'submission')` + `if not under_submission and st.st_mtime < artifact_mtime_floor`.
- `experiments/train_substrate_pretrained_driving_prior.py` calls `vendor_module_with_fresh_mtime` (4 occurrences) for module-body vendoring.

Catalog #270 dispatch optimization protocol: **9/9 PASS** via `tools/local_pre_deploy_check.py --strict --trainer experiments/train_substrate_pretrained_driving_prior.py --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch`. Tier 1 = 5/5, Tier 2 = 8/8, Tier 3 = 5/5.

## Pre-spawn-fatal observability event (Catalog #360 + #166)

First dispatch attempt at 17:11Z fired `pre_spawn_fatal` exit_code=2 per Catalog #166 + Catalog #360 PRE-spawn-fatal structured diagnostic. The diagnostic correctly classified 5 dirty paths:
- `.omx/state/canonical_equations_registry.jsonl` (sister state ephemeral)
- `.omx/state/modal_call_id_ledger.jsonl` (sister state ephemeral)
- `tools/build_hfv1_sparse_sidecar_candidate.py` (operator-owned source; mtime 03:08 UTC — 14h old; unrelated stale; sister-checkpoint guard PROCEED)
- `experiments/results/_modal_harvest_summary.json` (build artifact / derived output from my own harvest minutes earlier)
- `reports/cathedral_autopilot_evidence.jsonl` (unclassified)

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #230 ownership map + Catalog #340 sister-checkpoint guard, NONE of these files belong in MY commit scope. Per Catalog #202 paired-env bypass contract (`OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1` + `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1`), since (a) all 18 recipe-declared sentinel files VERIFIED CLEAN via `git diff --name-only` intersection, (b) Catalog #166 worker-side sentinel hash check STILL runs independently as fail-safe, (c) dirty paths categorized as out-of-scope per the diagnostic, paired-env bypass is the documented contract. Second dispatch attempt at 17:19Z succeeded.

## Dispatch state

| Arm | call_id | Dispatched at UTC | HEAD | Recipe | Expected wall-clock |
|---|---|---|---|---|---:|
| Baseline | `fc-01KS5RSNWQCYF5PR3KYPM8S9J9` | 2026-05-21T17:19:11Z | cd036aa610b9 | `substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch` | ~40 min |
| Procedural | `fc-01KS5RV15HVMFF39CHR2BJHKQ8` | 2026-05-21T17:20:00Z | cd036aa610b9 | `substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch` | ~40 min |

Both arms: T4 GPU, DPP_EPOCHS=25, timeout 2700s (45min), DPP_SKIP_AUTH_EVAL=1 (training-only emission; paired auth_eval fires separately).

Expected training cost per arm: ~$0.30 per recipe envelope; total ~$0.60 training + ~$0.40 for 4-arm paired auth_eval = ~$1.00 actual, within $2.00 operator envelope (1.7x slack on $1.20 expected per prompt).

Local lane dispatch dirs:
- `experiments/results/lane_substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_20260521T171911Z_modal/`
- `experiments/results/lane_substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_20260521T172000Z_modal/`

Modal app URLs (per Modal CLI output):
- baseline app: visible via `modal app list`
- procedural app: visible via `modal app list`

## Frontier-relevance per Catalog #316 + #343

Canonical PR 110 frontier pointer (read-only verification per Catalog #343 no-hardcoded-literals):
- `our_local_frontier_contest_cpu` = 0.192051 on archive `6bae0201...` (FEC6 fixed-Huffman k=16 clean; lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515; 178517 B; measured 2026-05-15T02:01:39Z)
- `our_local_frontier_contest_cuda` = 0.20533 on archive `9cb989cef519...` (PR106 format0d latent score-table; lane_pr106_format0d_latent_score_table_20260516; 186876 B; measured 2026-05-16T07:20:32Z)

DP1 substrate-class shift goal: produce first paid CONTEST-AXIS anchor for `canonical_equation #26 IN-DOMAIN dp1_codebook_bytes` (currently has BYTE-AXIS anchor only per OVERNIGHT-Z-RESUME `a1625378f` HARD-EARNED 2σ). Per recipe `predicted_band: [0.175, 0.190]`, DP1 paired CPU expected to beat FEC6 frontier by ~+0.005 if architectural shift hits OR underperform if substrate-class-shift hypothesis falsified. `reports/latest.md` will be updated per Catalog #316 ONLY after harvest + paired auth_eval completes.

## Operator-routable follow-up (priority-ordered)

### 1. Schedule harvest cron at T+45 min (2026-05-21T18:05Z = ~13:05 CDT)

The training arms run synchronously to ~40 min wall-clock; harvest cron at T+45min has safety margin. Cron command (operator-runnable from main thread):

```bash
.venv/bin/python tools/harvest_modal_calls.py --call-id fc-01KS5RSNWQCYF5PR3KYPM8S9J9 --execute
.venv/bin/python tools/harvest_modal_calls.py --call-id fc-01KS5RV15HVMFF39CHR2BJHKQ8 --execute
```

Expected outcome (Catalog #339 fail-closed registration ensures these outcomes land in `modal_call_id_ledger.jsonl`):
- HIGH verdict: rc=0 + 23+ artifacts + vendored submission_dir bodies PRESENT (verify via `find $DIR/harvested_artifacts/output/submission/src/ -name '*.py' -size +0c | wc -l` → should be 8+ non-zero `.py` files)
- LOW verdict: rc!=0 OR vendored bodies still empty (would require deeper GG fix investigation; sister #361 strict gate would have refused emission if regression occurred)

### 2. Fire 4-arm paired auth_eval AFTER training harvests

Once both training arms harvest with verified vendored bodies, fire 4-arm paired auth_eval via canonical helper per Catalog #246:

```bash
# Baseline paired auth_eval (CUDA + CPU on same archive)
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00 \
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive-zip experiments/results/lane_substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_20260521T171911Z_modal/harvested_artifacts/output/archive.zip \
    --inflate-sh experiments/results/lane_substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_20260521T171911Z_modal/harvested_artifacts/output/submission/inflate.sh \
    --lane-id lane_dp1_baseline_post_jj_paired_anchor_20260521 \
    --execute

# Procedural paired auth_eval (CUDA + CPU on same archive)
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00 \
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive-zip experiments/results/lane_substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_20260521T172000Z_modal/harvested_artifacts/output/archive.zip \
    --inflate-sh experiments/results/lane_substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_20260521T172000Z_modal/harvested_artifacts/output/submission/inflate.sh \
    --lane-id lane_dp1_procedural_post_jj_paired_anchor_20260521 \
    --execute
```

(Adjust paths once harvest dir structure confirmed — `tools/dispatch_modal_paired_auth_eval.py --help` for exact flag names; the canonical helper may have its own archive-discovery convention.)

### 3. Canonical equation #26 anchor registration (if HIGH verdict)

If 4-arm paired auth_eval produces contest-axis scores, register NEW empirical anchor on `canonical_equation #26 IN-DOMAIN dp1_codebook_bytes` per CLAUDE.md "Canonical equations + models registry":

```bash
.venv/bin/python -c "
from tac.canonical_equations import register_empirical_anchor, find_equation_by_id
eq = find_equation_by_id('procedural_codebook_from_seed_compression_savings_v1')
# Register anchor with axis_tag=[contest-CUDA] or [contest-CPU] per Catalog #323 Provenance
"
```

Per OVERNIGHT-Z-RESUME `a1625378f`, the current BYTE-AXIS anchor is `HARD-EARNED 2σ`. Adding CONTEST-AXIS anchor would be the first paid contest-axis anchor for DP1 substrate-class shift.

### 4. Sister coordination

Active sister at landing: NSCS06 v8 dispatch `fc-01KS5QRXWNVYC54E2Y9Z8KZ4W2` (cron 8a50fe12 will harvest at ~13:46 CDT per OVERNIGHT-QQ memo). DISJOINT substrate; no overlap with DP1 paths.

Other in-flight: 14 stale subagent checkpoints in `subagent_progress.jsonl` (Catalog #206 doesn't auto-mark-complete on natural session end). None touch DP1 paths.

## Cost summary

| Item | USD |
|---|---:|
| Baseline training (T4, 25ep, 2700s) | ~0.30 |
| Procedural training (T4, 25ep, 2700s) | ~0.30 |
| Subtotal training (this landing) | ~0.60 |
| 4-arm paired auth_eval (deferred to T+45min) | ~0.40 |
| **Total estimated** | **~1.00** |
| Operator-authorized envelope | 2.00 |
| Spent at landing | ~30% of budget |

## Discipline declarations

- Catalog #110/#113: APPEND-ONLY HISTORICAL_PROVENANCE — prior OVERNIGHT-CC + GG + Z-RESUME memos + canonical_equation #26 anchors PRESERVED unchanged; only NEW landings here
- Catalog #117/#157/#174: commit via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` (next step)
- Catalog #131/#138: fcntl-locked JSONL discipline + strict-load — modal_call_id_ledger appends + active_lane_dispatch_claims rewrites all via canonical helpers
- Catalog #166: pre_spawn_fatal correctly classified dirty tree at 17:11Z attempt 1; resolved via #202 paired-env bypass on attempt 2
- Catalog #199: paired-env operator-authorize bypass `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00`
- Catalog #202: paired-env whole-tree-clean bypass `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1` + `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1` (sentinels verified clean; #166 worker-side hash still runs)
- Catalog #205: canonical inflate device-fork preserved (DP1 inflate.py routes through `select_inflate_device`)
- Catalog #206: 4 own checkpoints emitted (in_progress)
- Catalog #220: substrate L1+ scaffold operational mechanism declaration (DP1 marked impl_complete; canonical equation #26 active)
- Catalog #229: PV (read OVERNIGHT-CC + GG + Z-RESUME memos + 2 recipes + frontier pointer + 1140+ LOC source verification before dispatch)
- Catalog #245: canonical Modal call_id ledger 4-layer pattern — 2 NEW dispatched rows via `register_dispatched_call_id_fail_closed` (auto-emitted by operator_authorize.py dispatch chain per #339)
- Catalog #246: paired auth_eval canonical helper to be invoked post-training-harvest
- Catalog #270: 9/9 PASS via local pre-deploy harness BEFORE dispatch
- Catalog #287: every empirical claim carries axis + source tag (commit ref / call_id / file path)
- Catalog #313: probe outcomes ledger NOT registered (this is SUBSTRATE training dispatch, not a probe per Catalog #313 sister scope)
- Catalog #316: canonical PR 110 frontier pointer cited (read-only via `canonical_frontier_pointer.json`); `reports/latest.md` will be updated ONLY after contest-axis scores land
- Catalog #323: canonical Provenance — IN_FLIGHT entries tagged `score_claim=false`, `evidence_grade=null`, `axis_tag=[predicted; pending_post_training]`
- Catalog #339: silent-no-spawn extinction — operator_authorize → modal_train_lane uses `register_dispatched_call_id_fail_closed` for both arms
- Catalog #340: sister-checkpoint guard PROCEED (no overlap with active sisters)
- Catalog #343: NO hardcoded score literals (PR 110 baseline cited via canonical pointer file only)
- Catalog #344: canonical equation #26 IN-DOMAIN `dp1_codebook_bytes` anchor pending post-harvest registration (will be CONTEST-AXIS sister to existing BYTE-AXIS HARD-EARNED 2σ anchor)
- Catalog #360: PRE-spawn-fatal observability ACTIVE (fired structured diagnostic at 17:11Z attempt 1)
- Catalog #361: harvester under_submission bypass ACTIVE (HEAD cd036aa61 contains the fix)
- CLAUDE.md "Carmack MVP-first phasing": Step 1 FREE local pre-deploy harness PASS BEFORE paid dispatch; Steps 2-5 apply post-harvest per re-train output validation
- CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": 2 NEW active_modal_training_in_progress rows registered via canonical `tools/claim_lane_dispatch.py claim --force`
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": 4-arm paired (CUDA+CPU × baseline+procedural) plan; Modal T4 = CUDA 1:1; Modal CPU container Linux x86_64 = CPU 1:1
- CLAUDE.md "Apples-to-apples evidence discipline": paired CPU + CUDA against IDENTICAL archive bytes per harvested arm
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE": operator-routable cron at T+45min ensures within 24h TTL window
- CLAUDE.md "Strategic Secrecy": no public PR action; internal dispatch only

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map: N/A at dispatch time (post-harvest may produce per-pair gradient anchors via DP1's compress-time gradient extraction; routed through `tac.sensitivity_map.*` only if substrate provides anchor file)
2. Pareto constraint: N/A at dispatch time (post-harvest paired auth_eval verdict will feed Pareto polytope solver only if contest-axis scores land)
3. Bit-allocator hook: N/A at dispatch time (no per-byte sensitivity emission from training-only dispatch)
4. Cathedral autopilot dispatch hook: **ACTIVE** — 2 NEW `dispatched` events in `modal_call_id_ledger.jsonl` consumable by autopilot's cost-band posterior for future DP1 dispatch ranking
5. Continual-learning posterior: **ACTIVE** — fcntl-locked appends to `.omx/state/modal_call_id_ledger.jsonl` (2 dispatched events) + `.omx/state/active_lane_dispatch_claims.md` (2 active_modal_training_in_progress rows)
6. Probe-disambiguator: **ACTIVE** — the dispatch verdict (HIGH = vendored bodies preserved + contest-axis scores; LOW = #361 fix didn't reach runtime) IS the disambiguator between (a) GG #361 fix is structurally complete vs (b) deeper META-fix needed; sister to Catalog #307 paradigm-vs-implementation discipline (this run validates the META-INFRASTRUCTURE fix actually works at the harvest layer, NOT a paradigm test of DP1)

## Files touched

- `.omx/state/modal_call_id_ledger.jsonl` (2 `dispatched` event rows appended via `register_dispatched_call_id_fail_closed`)
- `.omx/state/active_lane_dispatch_claims.md` (2 `active_modal_training_in_progress` rows registered via `tools/claim_lane_dispatch.py claim --force`)
- `.omx/state/subagent_progress.jsonl` (4 own checkpoint rows)
- `experiments/results/lane_substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch_20260521T171911Z_modal/modal_metadata.json` (dispatched; awaiting harvest)
- `experiments/results/lane_substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch_20260521T172000Z_modal/modal_metadata.json` (dispatched; awaiting harvest)
- `.omx/research/overnight_jj_dp1_re_train_plus_4_arm_paired_auth_eval_post_gg_361_fix_landed_20260521.md` (THIS memo)

## Lane

`lane_overnight_jj_dp1_re_train_plus_4_arm_paired_auth_eval_post_gg_361_fix_20260521` L1 (impl_complete-IN_FLIGHT pending T+45min harvest + 4-arm paired auth_eval; memory_entry).

Cost at landing: ~$0.60 baseline + procedural training (estimated; awaiting harvest cost confirmation per Modal billing dashboard) + ~$0 metadata writes + ~50 min wall-clock (PV + dispatch + Catalog #166 resolution + memo).
