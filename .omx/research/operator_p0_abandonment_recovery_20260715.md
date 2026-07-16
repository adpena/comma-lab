# OPERATOR-P0 ABANDONMENT RECOVERY — full ledger + the active-enforcement apparatus (2026-07-15)

**Trigger (operator verbatim):** *"There have been multiple things I have designated as p0 you have
abandoned because you forgot and silently got distracted and moved on to other things."* +
*"Do we need a hook or gate or something to remind you if p0 to survive compaction? That also
demands update when complete or when new p0 designated."*

**STORES CONSULTED:** the harness task ledger (~/.claude/tasks/89ff112f…, 488 tasks: 412 completed /
51 pending / 25 in_progress), `.omx/state/canonical_task_status.jsonl` (67 legacy tasks),
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (27 operator-P0 hits),
`.omx/research/P0_campaign_queue_20260715.md`, memory index MEMORY.md, recent 07-13..07-15 memos,
run artifacts (C0 launch.sh, v9_cgauge_432 run dir), trainer argparse, git log.

**Method:** enumerate every operator-designated P0 (tight-pattern task-ledger sweep + DAG grep +
memo sweep) → dedup to distinct directives → determine TRUE status by ARTIFACT EVIDENCE, never the
task checkbox (NO-FAKE "tasks marked done must actually be done") → rank recoverables → seed the
new canonical ledger `.omx/state/operator_p0_ledger.jsonl`.

## THE COUNT

**48 distinct operator-P0 directives** enumerated and seeded into the ledger:
- **26 GENUINELY-COMPLETE** (artifact-verified; incl. the 95%-kill wave #454–#465, the backward wave
  #486/#487, #500–#506 07-14 wave, #360/#378/#382/#395/#403–#407/#411/#421/#423/#424/#431).
- **8 PENDING-DROPPED / ABANDONED** (the operator's target class — see ranked table below).
- **7 STALLED-IN-PROGRESS** (real artifacts exist but the designated finish is owed: #497, #449,
  #349, #328, #377, #425, #343).
- **7 ACTIVE 07-15 open P0s** (not abandoned; registered so they structurally CANNOT be: #507, #509,
  #511, #512, #513, #514, #515).

Notable **marked-done-but-owed-followthrough** cases folded into the open rows: #404's child #408
telemetry batch (missed its boundary), #482's named ANE reactivation (VJP parity — no follow-up
artifact), #502's decisive curvelet A/B (tracked under #497).

## RANKED RECOVERY LIST (operator-emphasis × sub-0.15 impact; exact next action each)

| # | p0_id (task) | verbatim ask (date) | true status + evidence | recovery action | cost |
|---|---|---|---|---|---|
| 1 | p0_366_joint_pose_finishing (#366) | "Must pursue joint descent as a P0 in parallel with dseg polish" (07-08) | **PENDING-DROPPED 7 days.** Never launched. Partial mitigation: v7.5.2 curriculum carries the terminal joint pose-finish stage + banked R1 dxi 0.001610 fallback; the dedicated run pushing BELOW 0.001610 never fired. | Verify C0's pose-finish stage fires at its event trigger; else queue a dedicated Phase-2 finishing arm warm-started from C0 best ckpt. | launch-gated |
| 2 | p0_408_telemetry_resume_boundary (#408←#404) | land Q1–Q7 at the NEXT resume/stage boundary (07-10) | **ABANDONED AT ITS TRIGGER**: C0 launched 07-15 WITHOUT Q1–Q7 (H2 confound pass proved `--verdict-live-gap-every` ABSENT → EMA-lag sentinel missing, only L3 read-discipline covers it). | Land Q1–Q7 now (patch designs in telemetry_enhancement_audit_v7x_v8_20260710.md) so the NEXT launch carries them; Q3 as a DSL Lever. | $0 |
| 3 | p0_448_fresh_init_never_fired (#448) | FreSh init — "$0, MLX-trivial, **fire-this-run**" (07-12) | **BUILT-NEVER-FIRED** (`--fresh-init` at trainer:14169, default=False; C0 launched without it). The exact default-off orphan class. | $0 n24 A/B epochs-to-quality; fold ON into the next Phase-2 arm if it wins. | $0 |
| 4 | p0_444_bank_v9_stage_rows (#444) | byte-close each v9 stage EMA as it lands; queue exact-eval (07-11) | **PENDING-DROPPED**: stage ckpts EXIST (v9_cgauge_432 stageOctave1_ep251) but ZERO byte-closed/banked rows — even the $0 half never ran (only Modal was HOLD-gated). | $0 byte-close each landed stage EMA via tools/levelset_byte_close_and_eval.py; stage the exact-eval queue. | $0 (Modal=GO) |
| 5 | p0_496_madam_low_precision_rate (#496) | operator paper M+Adam as a RATE lever (07-13) | **PENDING-DROPPED**: two $0 probe axes designed, never fired. | $0 n600 axis-1 probe: fp4-train vs post-hoc-KKT-fp4 argmax survival at fixed bytes. | $0 |
| 6 | p0_485_jepa_latent_surrogate_dig (#485) | JEPA-latent surrogate for the cheap costate VJP (95%-kill family, 07-13) | **PENDING-DROPPED**: grounding landed 07-14 (arXiv 2605.27734 PRO), dig never fired. | $0: extract the poly-in-latent-dim bound + harvest RHM OSS + verdict. | $0 |
| 7 | p0_452_tube_algebra_rate_probe (#452) | "sol xhigh read+reference+deeply understand" 2607.07786 → measured per-boundary byte delta (07-12) | **PENDING-DROPPED**: no memo, never fired. | Fire the $0 codex sol arm → measured byte delta or formulation-scoped NO-GO. | $0 |
| 8 | p0_482_ane_vjp_parity_reactivation (#482) | "aggressively pursue ANE unlock" — the ladder's NAMED reactivation (07-13) | #482 itself honestly completed (verdict-advisory UNLOCKED_LOCAL banked); the named reactivation (VJP/cotangent parity + float32-native placement) has **NO follow-up artifact** — silently dropped post-harvest. | $0 local VJP-parity measurement; verdict decides the training-tier door. | $0 |
| 9 | p0_497_basis_cure_decisive_ab (#497+#502) | "Any potentially superior alternatives to Fourier features" (07-13) | **STALLED + ONCE-ORPHANED** (routed to an arm inbox after the arms exited). #502 BUILT the genuine curvelet/shearlet frames; the DECISIVE n600 through-R A/B (curvelet_through_R_dseg_ab) still owed — also gates the no-Fourier strict flip. | Fire curvelet_through_R_dseg_ab in its Phase-2 slot vs C0. | Phase-2 slot |
| 10 | p0_449_frozen_segnet_necessity (#449, operator-P0 ×3) | "does our stack need the frozen segnet fwd+bwd? optimal? alternatives?" (07-12) | **ARTIFACT-LANDED, NOT CLOSED**: 35.6K memo + the whole 95%-kill wave spawned from it; the in-loop timer confirmation (GO packet) owed to settle the contradictory fwd/bwd shares. | Close-out verdict row + fire GO_PACKET_p0_backward_k2_inloop_timer on operator GO. | $0 + GO |
| 11 | p0_425_phase_carrier_endgame (#425) | [P0] store-side phase codec end-to-end (07-11) | **BUILT+WIRED** (phase_residual_carrier.py in byte-close path) — measured end-to-end row owed. | $0 byte-close w/ phase-carrier mode on a cached witness output. | $0 |
| 12 | p0_328_clip_profile_rewire (#328) | auto-measured per-clip configs "rather than orphaned tools" (07-08) | Phase-1 BUILT; Phase-2 consumer rewire + Phase-3 60-tool fold **not started**. | Phase-2 at the next-launch boundary, measured-no-regression. | $0 |
| 13 | p0_349_control_theory_research (#349) | fields-medal CT vs the full stack (07-07) | **PARTIAL**: Pontryagin line sealed (#426/#433 measured); CT-2 synthesis unlanded. | Complete CT-2 synthesis or fold into #507/#515 with a disposition row. | $0 |
| 14 | p0_377_build_all_unbuilt (#377/#386) | "Build all the unbuilt" (07-09) | **SUBSTANTIALLY DELIVERED**; remainder (HORIZON-iso, STEP-iso fires, #220, #276) mapped to Phase-2 treatment slots. | Drain via Phase-2; keep activation-ledger rows honest. | Phase-2 |
| 15 | p0_343_dashboard_engineering (#343) | dashboard "broken but in need of engineering and optimization" | Long-running epic; refresh-latest landed; WebGPU remainder open. | Next increment on the single dashboard thread. | $0 |

Active-new 07-15 rows (#507 C1 cohesive, #509 burn-down, #511 clobber-gate+MEMORY-trim, #512 spawn
guard, #513 Modal single-flight, #514 white-box, #515 FINAL-OPTIMAL) are registered as
open/in_progress with their next actions — they are LIVE, and the ledger now makes silent dropping
structurally loud.

Also verified NOT abandoned (named triggers): #400/#406 (fire on first converged ckpt — Phase-3),
#380 crucible-3 (fires on crucible-2 P7), #444's Modal half (operator HOLD), #486/#487 in-loop timer
(OPERATOR-GO packet staged), #445 CUDA throughput (active — r6 harvested 9a582ced71, memo landed).

## THE STRUCTURAL FIX (two-landing; operator-demanded active enforcement)

**Root cause** (matches `dag_was_meant_to_be_reconstructable_graph_memory` L-row: WRITES>READS): P0
designations lived in chat context + a 488-row task ledger nobody re-reads; compaction erased them;
nothing DEMANDED status updates. The task checkbox also lies (NO-FAKE) — evidence was never re-verified.

**Landing 1 — the ledger + compaction-survival digest:**
- `.omx/state/operator_p0_ledger.jsonl` — canonical fcntl-locked append-only latest-wins ledger
  (schema: p0_id · designated_date · verbatim_ask · status ∈ open|in_progress|complete|superseded ·
  evidence · next_action · last_verified_utc · source · watch_paths? · task_ids?). COMMITTED
  (gitignore exception), seeded with all 48 rows above.
- `tools/operator_p0_digest.py` — ledger library + digest CLI. **Wired into the SessionStart hook
  chain (`.claude/settings.json`, no matcher ⇒ fires on startup/resume/clear/COMPACT)** alongside
  costate_digest.py: every fresh or compacted context window re-injects all open/in_progress
  operator-P0s with their NEXT action. This beats the goldfish problem: the apparatus remembers.
  Update path: `tools/operator_p0_digest.py --update <p0_id> --status … --evidence … --next-action …`
  (inherit-prior semantics; history preserved).

**Landing 2 — the Stop-hook demand-update (self-protection):**
- `tools/operator_p0_stop_hook.py` — warn-grade, fail-open, loop-safe Stop hook (modeled on
  tools/triality_drift_detector.py), wired into the Stop chain. Nags ONCE when:
  (A) the turn's commits touched a tracked open P0 (p0_id in subject, bound `#task` number, or
  watch_path prefix) but no ledger row was appended; (B) a NEW operator message designates a P0
  (word-bounded "p0" + directive verb) with no ledger row since. Escape valve: `[p0-ledger-ok]`
  commit token. Incremental transcript scanning (marker offsets) — old messages never re-fire.
- Tests: `src/tac/tests/test_operator_p0_apparatus.py` (24 tests: schema fail-closed writes,
  latest-wins, digest formatting, touched-P0 matchers, designation heuristics incl. word-boundary /
  tool-result / self-reference silence, fail-open integration smokes). All pass; ruff clean.

**Wire-in declaration (Catalog #125):** sensitivity-map N/A (apparatus) · Pareto N/A · bit-allocator
N/A · cathedral/autopilot: SessionStart+Stop hook chain IS the consumer · continual-learning: the
ledger is the posterior · probe-disambiguator N/A. verdict_scope: instance — every negative above is
an evidence-based status call on ONE directive, not a family claim.

Pointer 0.19108 submittable / 0.18804 borrowed-bank UNMOVED — this landing is apparatus
(anti-forgetfulness), honestly MEANS not ends; its value is that the 8 recovered P0s now each carry
an exact next action and can no longer silently vanish.
