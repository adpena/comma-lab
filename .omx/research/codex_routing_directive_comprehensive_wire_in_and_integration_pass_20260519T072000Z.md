# Codex routing directive — comprehensive wire-in + integration pass

**Date:** 2026-05-19T07:20:00Z
**Authority:** Operator-frontier-override per Catalog #300 + operator verbatim 2026-05-19 *"Wire in all and integrate all including the outstanding wire in cables, build and prep anything necessary too"* (sister to earlier same-session quote *"All operator decisions are approved, proceed with all and keep the queue saturated"*).
**Cumulative session budget envelope:** $30.50-60.30 USD paid GPU (per sister `codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md`) + $0 editor for wire-in work in THIS directive

## Operator-frontier-override frontmatter

```yaml
council_tier: T1  # working-group routing; binding decisions covered by operator master approval
council_attendees: [Claude-main-dispatcher, Codex-autonomous-research-and-implementation-loop]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_predicted_mission_contribution: frontier_protecting  # wire-in work IS frontier-protecting per CLAUDE.md "Subagent coherence-by-default" 6-hook discipline
council_override_invoked: true
council_override_rationale: "Wire in all and integrate all including the outstanding wire in cables, build and prep anything necessary too"
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: 2026-06-18T07:20:00Z
```

## Producer → consumer loop closure inventory

The session has 6 in-flight subagents producing infrastructure + 2 just-dispatched (HF dataset prep + master-gradient parser extension). The wire-in cables below close the producer→consumer loops AFTER the in-flight producers land. Codex picks up in dependency order.

### Wire-in #1: bit_allocator per-pair sensitivity consumer (TaskCreate #800)

**Depends on:** master-gradient parser extension (just dispatched; producer surface) + slot 6 per-byte sensitivity cathedral consumer (in-flight; reference pattern).

**Action:** Create NEW cathedral consumer package at `src/tac/cathedral_consumers/bit_allocator_per_pair_consumer/__init__.py` (~80-100 LOC mirroring the canonical Cable D consumer template). Required surface per Catalog #335 `CathedralConsumerContract` Protocol:
- `CONSUMER_NAME = "bit_allocator_per_pair_consumer"`
- `CONSUMER_VERSION = "1.0"`
- `CONSUMER_HOOK_NUMBERS = (3,)` (bit-allocator surface specifically per Catalog #125)
- `update_from_anchor` reads per-pair master-gradient anchors via `tac.master_gradient_consumers.load_master_gradient_for_archive`
- `consume_candidate` returns `ConsumerVerdict` with `predicted_delta_adjustment=0.0` (observability-only per Catalog #287/#323) + `axis_tag=[predicted]` + `promotable=False` + bit-allocator hint in `notes` field (top-K per-byte indices ranked by absolute sensitivity, suitable for `tac.bit_allocator.allocate_per_pair` if/when that helper exists)

**Tests:** `src/tac/tests/test_bit_allocator_per_pair_consumer.py` (~15 tests) mirroring slot 6's `test_per_byte_sensitivity_consumer.py` exactly with substituted hook number.

### Wire-in #2: cathedral_autopilot per-byte sensitivity predicted_dispatch_risk reweight (TaskCreate #801)

**NOTE:** With Catalog #335 auto-discovery paradigm, this is NOT a manual cathedral_autopilot edit — it's automatically handled by slot 6's `per_byte_sensitivity_consumer` (which sets `CONSUMER_HOOK_NUMBERS = (1, 3, 4)` including hook #4 cathedral autopilot dispatch). Verify slot 6's landing produces the canonical consumer that, when auto-discovered, IS the cathedral_autopilot reweight wire-in. NO additional codex work needed for #801 IF slot 6's consumer is contract-compliant.

If verification fails, file the gap as a follow-up Atom + write a 1-paragraph diagnostic memo at `.omx/research/cathedral_autopilot_per_byte_reweight_verification_20260519.md`.

### Wire-in #3: per-pair difficulty atlas → continual-learning posterior (TaskCreate #802)

**Action:** Create NEW cathedral consumer at `src/tac/cathedral_consumers/per_pair_difficulty_atlas_consumer/__init__.py` mirroring slot 6 pattern. Hook #5 ACTIVE (continual-learning posterior). Reads per-pair sensitivity from master-gradient ledger; computes per-pair "difficulty score" (`difficulty_p = ‖g_p‖ * pair_score_p` per Bayesian-experimental-design lens); emits continual-learning anchor row per-pair via `tac.continual_learning.posterior_update_locked` (Catalog #128). The anchor is `[predicted]` not `[empirical]` — difficulty atlas is a routing signal not a score claim.

**Tests:** `src/tac/tests/test_per_pair_difficulty_atlas_consumer.py` (~15 tests). 4-proc spawn-pool stress on continual-learning posterior write.

### Wire-in #4: master-gradient wire-in audit across ALL analytical surfaces (TaskCreate #890)

**Background:** Per sister synthesis memo, master-gradient is currently 47% under-wired across the 6 hooks. The just-dispatched master-gradient parser extension closes the per-archive producer side. This audit closes the per-consumer wire-in side across ALL existing canonical surfaces.

**Action:** Run `Bash(grep -rln "load_master_gradient_for_archive\|MasterGradientAnchor\|master_gradient" src/tac/ tools/ experiments/ --include="*.py")` to enumerate existing master-gradient touchpoints. For each touchpoint, classify per the 6-hook framework (Catalog #125). Write audit ledger at `.omx/research/master_gradient_wire_in_audit_v2_20260519.md` documenting current coverage % per hook + the gaps. Where Catalog #335 auto-discovery already closes the gap (a sister consumer package exists), mark CLOSED. Where a manual surface is needed, file as a follow-up Atom.

### Wire-in #5: HF Jobs dispatcher + cathedral_autopilot dispatch surface (TaskCreate #878)

**Depends on:** HF dataset prep subagent (just dispatched; produces the dataset + canonical HF Jobs training script template + dispatcher helper).

**Action:** After HF dataset prep lands, extend `tac.deploy.modal.call_id_ledger` (or sister `tac.deploy.hf_jobs.job_id_ledger` per the dataset prep subagent's design) to accept `platform="hf_jobs"` for ledger registration parity. Then create NEW cathedral consumer at `src/tac/cathedral_consumers/hf_jobs_dispatcher_consumer/__init__.py` that surfaces HF Jobs as a routing target candidate (alongside Modal / Lightning / Vast.ai / local-mps / local-cpu). Hook #4 ACTIVE.

### Wire-in #6: paired dispatch ranker via HF Jobs free-tier exploitation

**Background:** HF Jobs T4-small at $0.40/hr is competitive with Modal T4 ($0.59/hr) on cost AND the canonical training script template is well-tested. Cathedral autopilot routing should consider HF Jobs as a first-class target.

**Action:** Extend `tac.cost_band_calibration.PLATFORM_RATES_USD_PER_HOUR` dict to include HF Jobs flavors (t4-small=$0.40 / l4x1=$0.80 / a10g-large=$1.50 / a100-large=$3.50 per the canonical HF Jobs pricing table in the `hugging-face-jobs` skill). Add to `_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS` for the `eval` dispatch class (HF Jobs T4 is 32% cheaper than Modal T4 for eval workloads where shared-volume FS contention isn't a factor).

## Build + prep items (codex follow-on after slot wave lands)

### Build #1: Catalog #523 L2 Hinton-distilled SegNet surrogate Phase 1 BUILD (TaskCreate #875)

**Depends on:** HF dataset prep (just dispatched).

**Action:** After dataset lands at `adpena/comma-video-segnet-image-level-600pairs`, dispatch the HF Jobs training run via `tools/dispatch_hf_jobs_vision_training.py --dataset adpena/comma-video-segnet-image-level-600pairs --model timm/mobilenetv3_small_100.lamb_in1k --hub-model-id adpena/comma-segnet-surrogate-mobilenetv3-distilled --num-train-epochs 30 --metric-for-best-model eval_accuracy`. Cost cap $5 (well within session envelope). Per Catalog #325 6-step contract: write per-substrate symposium memo BEFORE dispatch.

### Build #2: Z6-v2 Wave 2 4c re-fire AFTER silent-no-spawn fix lands (TaskCreate parent of #674)

**Depends on:** slot 1 silent-no-spawn structural extinction (in-flight).

**Action:** When slot 1 lands its 4-layer fix (canonical helper + STRICT gate + sister mitigation + tests) + the post-dispatch ledger poll catches the 3rd-consecutive failure pattern, re-fire Z6 Wave 2 4c via `bash scripts/operator_authorize_substrate_time_traveler_l5_z6_modal_a10g_dispatch.sh` (already operator-approved per `feedback_z6_v2_wave_2_codex_repairs_landed_20260517.md` operator-frontier-override). Cost cap $3 per recipe envelope.

### Build #3: STC v2 RATIFY-or-DEFER (TaskCreate parent of #769)

**Depends on:** slot 1 silent-no-spawn fix.

**Action:** After unblock, re-fire STC v2 smoke via `bash scripts/operator_authorize_substrate_stc_v2_modal_t4_dispatch.sh`. Register verdict in `.omx/state/probe_outcomes.jsonl` per Catalog #313. Cost cap $0.20.

## Discipline (uniform across all items)

- **One commit per item** via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #117/#157/#174/#235
- **Catalog #229 PV** (premise verification) BEFORE every edit: GREP for cited canonical names via `importlib.util.find_spec()` or `Bash(grep)` to verify the canonical helper actually exists at the claimed path
- **Catalog #206 checkpoint discipline** for every long-running task (>5 tool uses estimate)
- **Catalog #230 sister-subagent ownership map**: declare scope BEFORE editing shared surfaces (preflight.py / CLAUDE.md / operator_authorize.py); coordinate via --expected-content-sha256 with rc=4 retry-and-rebase
- **Catalog #313 probe-outcomes ledger** registration mandatory for every dispatch verdict (PROCEED / DEFER / INDEPENDENT / KILL / PARTIAL / OPERATOR_REVIEW_REQUIRED)
- **Catalog #314 absorption-pattern PREVENTION** (currently in-flight via slot 3): use only `tools/subagent_commit_serializer.py`; NEVER bare `git add` + `git commit`; check sister checkpoint via canonical helper after slot 3 lands
- **Catalog #325 per-substrate symposium evidence** within 14 days BEFORE any paid dispatch; OR operator-frontier-override per Catalog #300 (operator master approval covers this directive's items)
- **Per CLAUDE.md "Apples-to-apples evidence discipline"**: ALL scores must carry axis + hardware + archive sha tags; NEVER infer CPU from CUDA or vice versa; NEVER score-claim from advisory-grade signals (MPS / macOS-CPU / proxy / partial)
- **Per CLAUDE.md "Forbidden premature KILL without research exhaustion"**: DEFER + REQUEST-REINVESTIGATION-OF-ALTERNATIVES per Catalog #308 is the canonical verdict structure; NO KILL without grand council consensus + research exhaustion

## Maximum-signal preservation per Catalog #300

- Verbatim dissent: none on the routing itself
- Per-member operating-within assumption: HARD-EARNED via canonical Catalog #335 auto-discovery paradigm + Catalog #125 6-hook wire-in discipline + Catalog #327 master-gradient producer surface
- HARD-EARNED-vs-CARGO-CULTED classification per Wire-in:
  - Wire-in #1-3: HARD-EARNED (Cable I3+I4+I5 explicit in `integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md`)
  - Wire-in #4: HARD-EARNED-EMPIRICALLY-VERIFIED (sister synthesis identified 47% under-wired)
  - Wire-in #5+#6: HARD-EARNED-PER-OPERATOR-DIRECTIVE (operator standing approved HF Jobs path)
- Full vote tally: PROCEED-unconditional via operator-frontier-override
- Cite-chain: this directive + the integrated battle plan + the operator-frontier-override capture

## Cross-references

- Parent operator-frontier-override capture: `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
- Sister codex routing directives:
  - `codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md` (B+C+E.1+F batch)
  - `codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md` (paid C6 batch)
  - `codex_routing_directive_codex_to_claude_inbox_bidirectional_channel_20260518.md` (Catalog #333)
- Integrated battle plan: `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md`
- Slot 6 per-byte sensitivity consumer landing (in-flight; agent ID aa90c98a047d059f6): canonical pattern for Wire-ins #1+#3
- Master-gradient parser extension (in-flight; agent ID TBD from this turn's dispatch): producer surface for Wire-ins #1+#3+#4
- HF dataset prep (in-flight; agent ID TBD from this turn's dispatch): producer surface for Wire-in #5 + Build #1

— Claude-main 2026-05-19T07:20:00Z (comprehensive wire-in routing per operator master approval)
