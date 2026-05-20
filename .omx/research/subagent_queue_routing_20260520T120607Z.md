# Subagent Queue Routing — Task Triage — 2026-05-20T12:06:07Z

> Companion to `.omx/research/task_triage_inventory_20260520T120607Z.md`.
> Lists 2 SUBAGENT-routed tasks. **Operator approves SLOT dispatch order;
> this directive memo IS the canonical prompt for each.**

## Routing rationale (per task brief Rule 3)

Both SUBAGENT-routed tasks (#5 + #6 from triage inventory) qualify because:
- Contained scope (single subagent slot per task; ≤90min wall-clock each)
- Operator approval IMPLICIT per silent-no-spawn fix landing commit `233fce252` (Catalog #339) + the prior `paid_dispatch_batch_C6_plus_204_followon` directive operator-approved 2026-05-19
- Implementation/verification work (NOT design memos; NOT operator decisions)
- Post-fix verification of structural-extinction is the canonical next step per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"

## Slot collision check (sister-coordination per Catalog #340)

Current in-flight slots per `.omx/state/subagent_progress.jsonl`:
- mg15 / slot_mg_16 / slot_mg_17 / slot_mg_18 / mg19 (editorial wave, `docs/` scope)
- grand-council-t3-strategy-revi (THIS triage's parent)
- task-triage-20260520 (THIS subagent)

**Next available SLOT letters**: TT-1 (post-MG-19) / TT-2 (sister-paired with TT-1).

Proposed SLOT labels: `SLOT-TT-1-Z6-WAVE-2-4C-POST-FIX-VERIFICATION` + `SLOT-TT-2-STC-V2-RATIFY-OR-DEFER`.

---

## SLOT-TT-1 — Z6 Wave 2 4c re-fire (Triage task #5; sister-supersedes task #8)

**Estimated cost**: $3 Modal A10G smoke + $0 local
**Estimated wall-clock**: 60-90min total (5min local pre-deploy + 30-50min Modal smoke + 15-30min harvest + verification + commit)

**Canonical subagent prompt (~500 words)**:

```
ROLE: SLOT-TT-1 — Z6 Wave 2 4c re-fire post-silent-no-spawn-fix verification

CONTEXT: silent-no-spawn structural extinction landed commit `233fce252`
(Catalog #339) closing the bug class where `fn.spawn()` succeeded but
register_dispatched_call_id was wrapped in silent-swallow try/except,
leaking orphan paid Modal dispatches. Z6 Wave 2 4c is the canonical
verification target per the post-fix directive
`codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z::ITEM_5`
+ sister `comprehensive_wire_in_and_integration_pass::BUILD_2`.

PRE-FLIGHT (mandatory per Catalog #229):
1. Read CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + Catalog #339 +
   Catalog #245 modal_call_id_ledger
2. Read `tools/subagent_checkpoint.py` for crash-resume discipline (Catalog #206)
3. Read recent silent-no-spawn fix landing:
   `feedback_silent_no_spawn_structural_extinction_landed_20260519.md`
4. Read Z6 Wave 2 4c recipe:
   `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_a10g_dispatch.yaml`
   verify it has `Z6_TRAINER_MODE: "full"` + `SMOKE_ONLY: "0"` per Catalog #326
5. Read `.omx/state/probe_outcomes.jsonl` for any Z6-substrate blocking
   verdict (Catalog #313)
6. Run `tools/operator_authorize.py --recipe substrate_time_traveler_l5_z6 \
       --target modal --dry-run --strict` to verify pre-dispatch clean
7. Run `tools/local_pre_deploy_check.py --strict` per Catalog #243

EXECUTION:
1. Fire `tools/operator_authorize.py --recipe substrate_time_traveler_l5_z6 \
       --target modal --yes` (paid $3 Modal A10G; SMOKE only per recipe)
2. Capture call_id from stdout
3. Verify within 60s: `.venv/bin/python -c "
   from tac.deploy.modal.call_id_ledger import query_by_call_id
   rows = query_by_call_id('<call_id>')
   assert len(rows) >= 1 and rows[0]['event_type'] == 'dispatched'
   print('Catalog #339 + #245 wire-in verified: ledger row present')"`
4. Monitor for 30-50min via `tools/harvest_modal_calls.py --call-id <id>`
5. On completion: verify ledger has `harvested` event + score is finite +
   axis_tag is `[contest-CUDA]` or `[contest-CPU]` per recipe declaration

CHECKPOINT DISCIPLINE (Catalog #206):
- After each major milestone: `tools/subagent_checkpoint.py --subagent-id \
    slot-tt-1-z6-wave-2-4c-post-fix --step <N> --status in_progress \
    --files-touched <list> --next-action <next>`

POST-EXECUTION:
1. If smoke GREEN (rc=0 + finite score + ledger row clean): mark Z6 Wave 2
   4c PROCEED via tac.probe_outcomes_ledger.register_probe_outcome
2. If smoke RED: mark DEFER per Catalog #313 + classify per Catalog #307
   (paradigm-vs-implementation falsification)
3. Update canonical_task_status.jsonl: status=completed for BOTH
   `paid_dispatch_batch::ITEM_5` AND `comprehensive_wire_in_and_integration_pass::BUILD_2`
   (the latter is sister-superseded by this work)
4. Commit via tools/subagent_commit_serializer.py with
   --expected-content-sha256 (Catalog #157/#174 mandatory)
5. Write landing memo per Catalog #125 6-hook wire-in declaration:
   ~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_tt_1_z6_wave_2_4c_post_fix_verification_landed_<UTC>.md

DISCIPLINE:
- Catalog #117/#119/#157/#174/#206/#229/#287/#323 standard
- Catalog #245 + #339 wire-in verification IS the deliverable
- per CLAUDE.md "Modal .spawn() HARVEST OR LOSE": harvest within 24h MANDATORY
- per CLAUDE.md "Forbidden premature KILL": DEFER on red, not KILL

ROUTING: operator approves the $3 Modal spend BEFORE dispatch fires
(implicit via this routing memo OR explicit re-confirm).
```

---

## SLOT-TT-2 — STC v2 RATIFY-or-DEFER (Triage task #6; sister-supersedes task #9)

**Estimated cost**: $0.20 Modal T4 smoke + $0 local
**Estimated wall-clock**: 30-45min total (5min local pre-deploy + 5-10min T4 smoke + 10-15min harvest + verification + commit)

**Canonical subagent prompt (~400 words)**:

```
ROLE: SLOT-TT-2 — STC v2 RATIFY-or-DEFER post-silent-no-spawn-fix smoke

CONTEXT: Sister of SLOT-TT-1. STC v2 (Syndrome-Trellis Coding for per-frame
mask payload per Filler 2011 + Fridrich inverse-steganalysis) is the
canonical SECOND verification target post-Catalog #339 silent-no-spawn fix.
Per directives:
- `codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z::ITEM_6`
- `codex_routing_directive_comprehensive_wire_in_and_integration_pass_20260519T072000Z::BUILD_3`

PRE-FLIGHT (mandatory per Catalog #229):
1. Read CLAUDE.md NeRV-family parity discipline (HNeRV L1-13) + Catalog #339
2. Read `tools/subagent_checkpoint.py` for crash-resume (Catalog #206)
3. Read recent STC landing memos:
   `feedback_silent_no_spawn_structural_extinction_landed_20260519.md`
   sister STC trainer at `experiments/train_substrate_stc_v2.py`
4. Read STC v2 recipe:
   `.omx/operator_authorize_recipes/substrate_stc_v2_modal_t4_dispatch.yaml`
   verify driver-mode env-var compliance per Catalog #326
5. Read `.omx/state/probe_outcomes.jsonl` for any STC-substrate blocking verdict
6. Run pre-deploy clean check per Catalog #243

EXECUTION:
1. Fire `tools/operator_authorize.py --recipe substrate_stc_v2 \
       --target modal --yes` (paid $0.20 Modal T4)
2. Capture call_id + verify Catalog #245/#339 ledger row lands within 60s
3. Monitor for 5-10min via harvester
4. On completion: verify ledger + score axis-tag

POST-EXECUTION:
1. If smoke GREEN: register PROCEED verdict via
   tac.probe_outcomes_ledger.register_probe_outcome + RATIFY decision in
   landing memo
2. If smoke RED: register DEFER verdict per Catalog #313 +
   Catalog #307 paradigm-vs-implementation classification +
   Catalog #308 alternative-probe-methodology enumeration (per
   "Forbidden premature KILL")
3. Update canonical_task_status.jsonl: status=completed for BOTH
   `paid_dispatch_batch::ITEM_6` AND `comprehensive_wire_in_and_integration_pass::BUILD_3`
4. Commit via subagent_commit_serializer with --expected-content-sha256
5. Write landing memo per Catalog #125 6-hook wire-in declaration:
   feedback_slot_tt_2_stc_v2_ratify_or_defer_landed_<UTC>.md

DISCIPLINE: same as SLOT-TT-1 with explicit Catalog #307/#308 application
on RED outcome.

ROUTING: operator approves the $0.20 Modal spend BEFORE dispatch fires.
```

---

## Dispatch ordering recommendation

**Sequential (NOT parallel)** despite the 4-subagent parallel cap, because:
1. Both verify the SAME Catalog #339 silent-no-spawn fix → SLOT-TT-2 inherits SLOT-TT-1's verification confidence
2. SLOT-TT-1 is higher-EV ($3 unblocks ASYMPTOTIC-PURSUIT per T3 Decision 4 item 4 vs STC at lower band-shift potential)
3. Sequential prevents simultaneous Modal billing-meter contention
4. If SLOT-TT-1 RED → SLOT-TT-2 verification still valuable (proves the fix works for cheaper smokes)

**Recommended order**: SLOT-TT-1 first (60-90min) → SLOT-TT-2 second (30-45min).

**Total subagent wall-clock**: ~2-2.5h sequential.
**Total operator-routable cost**: $3.20 (Modal A10G + Modal T4).

## Summary

| SLOT | Task | Cost | Wall-clock | Output |
|------|------|------|------------|--------|
| TT-1 | Z6 Wave 2 4c re-fire | $3 | 60-90min | landing memo + ledger verification + Catalog #339 confidence anchor |
| TT-2 | STC v2 RATIFY-or-DEFER | $0.20 | 30-45min | landing memo + RATIFY-or-DEFER verdict + secondary Catalog #339 anchor |

**Both SLOTs together STALE-CLOSE 4 of the 12 active tasks** (#5+#6 directly + #8+#9 as sister-superseded).
