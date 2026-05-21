---
council_tier: T1
council_attendees: [Claude]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent: []
council_decisions_recorded:
  - "fired 4 paired Modal auth_eval dispatches on saved DP1 PATH A archives (baseline + procedural × CUDA + CPU); all 4 returned rc=1 in ~3s"
  - "diagnosed IMPLEMENTATION-LEVEL failure: DP1 trainer's _write_runtime emits empty __init__.py stubs but does NOT vendor module bodies (sister bug class to Catalog #295)"
  - "registered 4 canonical Modal call_id ledger rows + 4 outcome rows per Catalog #245 + #339 fail-closed discipline"
  - "closed 4 NEW lane dispatch claims as terminal failed; also closed 2 STALE active claims from prior OVERNIGHT-Z PATH A dispatch (canonical helper cleanup)"
  - "operator-routable: fix DP1 _write_runtime to vendor src/tac/substrates/pretrained_driving_prior/{inflate.py, prior_application.py, ...} into submission_dir/src/tac/... BEFORE re-firing auth_eval"
  - "operator-routable: HF Jobs RECHARGE is operator-only (must log into HF Jobs dashboard); surface in main thread"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - dp1_3rd_attempt_path_a_success_first_paid_byte_anchor_canonical_equation_26_registration_landed_20260521
---

# OVERNIGHT-CC DP1 PATH A auth_eval RE-FIRE on saved archives blocked by vendor-stub bug LANDED 2026-05-21

## Summary

Per operator authorization 2026-05-21 verbatim *"All are approved on my end"* + canonical PR 110 frontier reference baseline per Catalog #316, attempted to re-fire `experiments/contest_auth_eval.py` (paired CPU + CUDA on Modal Linux x86_64 per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE") on the saved DP1 PATH A archives from OVERNIGHT-Z-RESUME (commit `a1625378f`).

**All 4 dispatches failed rc=1 in ~3s with the SAME ModuleNotFoundError** — the saved `submission/inflate.py` imports `from tac.substrates.pretrained_driving_prior.inflate import inflate_one_video` but the vendored `src/tac/substrates/pretrained_driving_prior/__init__.py` (and all sister sub-packages) are EMPTY 0-byte stubs. The DP1 trainer's `_write_runtime` creates the directory tree but does NOT vendor the actual module bodies.

This is an IMPLEMENTATION-LEVEL bug class (sister of Catalog #295 `check_submission_inflate_works_with_empty_pythonpath`) per CLAUDE.md "Forbidden premature KILL" — NOT a paradigm-level falsification of DP1 substrate or auth_eval re-fire approach.

## Empirical observations

### Saved archives (verified bytes-identical to OVERNIGHT-Z-RESUME)

| Arm | Archive path | sha256 | Bytes |
|---|---|---|---|
| Baseline | `experiments/results/dp1_3rd_attempt_harvested_baseline_20260521/output/archive.zip` | `b5ac83d17d4a935564b7836c7534db40329f9de3dbd237fcaeeca2b2a8b10901` | 25733 |
| Procedural | `experiments/results/dp1_3rd_attempt_harvested_procedural_20260521/output/archive.zip` | `a2e52986d288f6e388b0a6708148395910670e9973ea0980551d0d5b2c41e6e6` | 18269 |

### 4 dispatched arms (Modal call_id ledger registered)

| Arm | call_id | rc | elapsed | Cost (USD) |
|---|---|---:|---:|---:|
| Baseline-CUDA | `fc-01KS5KW2EZBYYKV8DP1H8T0X2D` | 1 | ~3s | ~0.001 |
| Baseline-CPU | `fc-01KS5KWQZSADWJ98C7HXFEE117` | 1 | ~3s | ~0.001 |
| Procedural-CUDA | `fc-01KS5KXSXVP7RKB7CV2J37JJX0` | 1 | ~3s | ~0.001 |
| Procedural-CPU | `fc-01KS5KYD9SB1MT176QW5V1KFJN` | 1 | ~3s | ~0.001 |
| **Total** | | | ~12s | **~0.004** |

All 4 returned `validation_errors=["contest_auth_eval.json was not produced"]`. The procedural-CPU arm captured full stderr showing:
```
ModuleNotFoundError: No module named 'tac.substrates.pretrained_driving_prior.inflate'
```

## Root cause: submission_dir is structurally incomplete

Both `dp1_3rd_attempt_harvested_baseline_20260521/output/submission/` and the procedural sister directory contain:
- `inflate.sh` (377 B, Catalog #146 3-positional-arg contract — correct)
- `inflate.py` (1.1 KB stub importing from `tac.substrates.pretrained_driving_prior.inflate`)
- `0.bin` (the archive payload)
- `src/tac/__init__.py` (0 bytes)
- `src/tac/procedural_codebook_generator/__init__.py` (0 bytes)
- `src/tac/substrates/__init__.py` (0 bytes)
- `src/tac/substrates/_shared/__init__.py` (0 bytes)
- `src/tac/substrates/pretrained_driving_prior/__init__.py` (0 bytes)

The directory tree exists but the actual module bodies (`inflate.py` body / `prior_application.py` / `_shared/inflate_runtime.py` / `procedural_codebook_inflate.py` / etc.) are NEVER vendored into `src/tac/substrates/pretrained_driving_prior/` — only the empty `__init__.py` stubs. On Modal worker (`PYTHONPATH=""`) the import fails immediately.

The real source files exist at `src/tac/substrates/pretrained_driving_prior/`:
- `inflate.py` (8.8 KB)
- `prior_application.py` (11.9 KB)
- `procedural_codebook_inflate.py` (15.9 KB)
- `architecture.py` (5.4 KB)
- `archive.py` (16.5 KB)
- `_shared/inflate_runtime.py` (canonical helper)

The DP1 trainer's `_write_runtime` (in `experiments/train_substrate_pretrained_driving_prior.py` or `src/tac/substrates/pretrained_driving_prior/archive.py`) needs to ALSO copy these module bodies into the submission_dir.

## Sister bug class: Catalog #295

Catalog #295 `check_submission_inflate_works_with_empty_pythonpath` covers the SOURCE-TEXT surface — refuses inflate.py that imports `from tac.*` without vendored alongside. The DP1 submission_dir would FAIL Catalog #295 if scanned (it has `from tac.substrates.pretrained_driving_prior.inflate import ...` but only empty stub `tac/` packages without the actual module). However, Catalog #295's scope is `submissions/*/inflate.py` (NOT `experiments/results/**/output/submission/inflate.py` per the DERIVED_OUTPUT exclusion per Catalog #113).

This empirical anchor surfaces a NEW META-class: **trainer-emitted submission_dir bytes that pass local PYTHONPATH-aware tests but fail under Modal worker's empty-PYTHONPATH environment**. The bug is in the trainer's vendoring step, NOT the inflate.py itself.

## Cost summary

| Item | USD |
|---|---:|
| 4 Modal auth_eval arms × ~3s × ~$0.001 ea | ~0.004 |
| Operator-authorized budget envelope | 2.00 |
| Spent | 0.20% of budget |

## Frontier-relevance check per Catalog #316

Canonical PR 110 frontier pointer (read-only verification per Catalog #343):
- `our_local_frontier_contest_cpu`: 0.192051 on archive `6bae0201...` 178517 B (FEC6 fixed-Huffman k=16 clean)
- `our_local_frontier_contest_cuda`: 0.20533 on archive `9cb989cef519...` 186876 B (PR106 format0d latent score-table)

DP1 PATH A auth_eval produced NO contest-axis scores; no frontier comparison possible. `reports/latest.md` NOT updated.

## Operator-routable follow-up (priority-ordered)

1. **DP1 trainer fix**: edit `src/tac/substrates/pretrained_driving_prior/archive.py` or `experiments/train_substrate_pretrained_driving_prior.py::_write_runtime` to vendor the actual module bodies (copy `inflate.py`, `_shared/inflate_runtime.py`, `prior_application.py`, `procedural_codebook_inflate.py`, etc.) into `submission_dir/src/tac/substrates/...` (NOT just the empty stubs). Sister discipline: Catalog #295 + the canonical NSCS06-v6 vendor pattern (per CLAUDE.md "Forbidden /tmp paths" + Catalog #205 inflate device-fork).

2. **Re-fire auth_eval after fix**: re-execute today's canonical 2-arm paired-dispatch command after vendor-fix lands + retrain to land a NEW archive with the correct vendored submission_dir. Approximate cost: $0.30 per dispatch × 4 arms = $1.20 (well within operator's $2.00 envelope).

3. **HF Jobs RECHARGE** (operator-only action, cannot be done from Claude): operator must log into HF Jobs dashboard and recharge $5 to unblock HF-Jobs-budgeted lanes.

4. **NSCS06 v8 Phase 4** (queued for next slot rotation per cap=2 stagger discipline): operator authorization preserved for NSCS06 v8 dispatcher subagent when slot frees.

5. **Catalog #295 scope extension** (long-term): consider extending Catalog #295's gate scope to also scan `experiments/results/**/output/submission/inflate.py` artifacts to catch THIS bug class structurally at the trainer-output surface. Currently DERIVED_OUTPUT exclusion (Catalog #113) prevents this; the right answer is to instead add a NEW gate in the DP1 trainer's `_write_runtime` that REFUSES emission when vendored module bodies are absent (sister of Catalog #146 inflate runtime contract).

## Discipline declarations

- Catalog #206: predecessor checkpoint read (none) + 5 own checkpoints emitted
- Catalog #117/#157/#174: commit via canonical serializer with POST-EDIT `--expected-content-sha256` (next step)
- Catalog #110/#113: APPEND-ONLY HISTORICAL_PROVENANCE — only NEW rows on `modal_call_id_ledger.jsonl` + `active_lane_dispatch_claims.md`; PRIOR OVERNIGHT-Z-RESUME landing memo + canonical_equation #26 anchors PRESERVED unchanged
- Catalog #199: paired-env operator-authorize bypass `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00`
- Catalog #229: PV (read all relevant source + recipe + landing memo before dispatch)
- Catalog #245: canonical Modal call_id ledger 4-layer pattern — 4 NEW rows registered via canonical helper
- Catalog #287: NO docstring overstatement — all numbers tagged with axis + source
- Catalog #289: paired-env bypass properly observed
- Catalog #313: probe outcomes ledger NOT registered (this is a SUBSTRATE dispatch failure, not a probe; per Catalog #313 sister scope)
- Catalog #316: canonical PR 110 frontier pointer cited (read-only via canonical_frontier_pointer.json); `reports/latest.md` NOT updated (no contest-axis scores landed)
- Catalog #323: canonical Provenance discipline — call_id ledger rows tagged `evidence_grade=null` (failed dispatch; no score claim)
- Catalog #339: silent-no-spawn extinction — used `register_dispatched_call_id_fail_closed` for all 4 arms
- Catalog #340: sister-checkpoint guard active (canonical serializer wire-in)
- CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": 2 stale active claims from earlier OVERNIGHT-Z dispatch closed via `tools/claim_lane_dispatch.py --force` (terminal_harvested_rc_0) BEFORE NEW dispatch fired
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE": paired CPU + CUDA dispatched; both Modal Linux x86_64 = 1:1 contest-compliant per the canonical helper's contract
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE": harvest performed within the same session via `tools/recover_modal_auth_eval.py` for all 4 arms; result.json files saved to `experiments/results/modal_auth_eval{,_cpu}/dp1_path_a_*/`

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map: N/A (no per-element sensitivity update; failed-dispatch artifacts contain no score signal)
2. Pareto constraint: N/A (no scorer distortion measured)
3. Bit-allocator: N/A (no rate-axis signal from failed dispatch)
4. Cathedral autopilot dispatch hook: ACTIVE — failed-rc=1 outcome rows in `modal_call_id_ledger.jsonl` are consumable by autopilot's cost-band posterior so future ranker decisions can de-prioritize DP1 auth_eval until vendor-fix lands
5. Continual-learning posterior: ACTIVE via fcntl-locked append to `.omx/state/modal_call_id_ledger.jsonl` (4 dispatched events + 4 failed outcomes); `.omx/state/active_lane_dispatch_claims.md` (4 NEW terminal rows + 2 stale-close rows)
6. Probe-disambiguator: ACTIVE — the empirical failure mode (vendor-stub-without-module-body) IS the disambiguator between (a) DP1 substrate paradigm vs (b) DP1 trainer's `_write_runtime` IMPLEMENTATION-level bug per Catalog #307 paradigm-vs-implementation classification

## Sister coordination

Per CLAUDE.md "Subagent coherence-by-default": 14 in-flight subagent checkpoints in last 2h (most stale from completed work — Catalog #206 doesn't auto-mark-complete). The ACTIVE sister per operator stagger discipline = Slot 1 OVERNIGHT-AA STC 3b probe (DISJOINT — touches `tools/probe_stc_3b_*` + Selfcomp source READ-ONLY). NO collision risk; THIS subagent touched only NEW files + APPEND-ONLY canonical state stores.

## Files touched

- `.omx/state/modal_call_id_ledger.jsonl` (4 dispatched + 4 failed-outcome rows appended via canonical helper)
- `.omx/state/active_lane_dispatch_claims.md` (4 NEW terminal_failed rows + 2 stale-close terminal_harvested_rc_0 rows)
- `.omx/state/subagent_progress.jsonl` (5 own checkpoint rows)
- `experiments/results/modal_auth_eval/dp1_path_a_baseline_refire_paired_modal_auth_20260521T155324Z_cuda/` (artifacts harvested)
- `experiments/results/modal_auth_eval_cpu/dp1_path_a_baseline_refire_paired_modal_auth_20260521T155324Z_cpu/` (artifacts harvested)
- `experiments/results/modal_auth_eval/dp1_path_a_procedural_refire_paired_modal_auth_20260521T155424Z_cuda/` (artifacts harvested)
- `experiments/results/modal_auth_eval_cpu/dp1_path_a_procedural_refire_paired_modal_auth_20260521T155424Z_cpu/` (artifacts harvested)
- `.omx/research/overnight_cc_dp1_path_a_auth_eval_refire_blocked_by_vendor_stub_bug_landed_20260521.md` (THIS memo)

## Lane

`lane_overnight_cc_dp1_path_a_auth_eval_refire_on_saved_archives_pr110_reference_20260521` L1 (impl_complete-DEFER_PENDING_EVIDENCE: empirical anchor that DP1 submission_dir vendor-fix is REQUIRED-prereq before any auth_eval re-fire can produce contest-axis scores; memory_entry).

Cost: ~$0.004 actual paid GPU + ~45 min wall-clock (PV + dispatch + harvest + landing memo).
