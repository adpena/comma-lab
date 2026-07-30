# ddm_hw1 — MAIN COGNITION / HARNESS WAVE (task #785, ledger QA88)

**POINTER HONESTY FIRST:** `0.1910828242 [contest-CPU custody]` **UNMOVED**. Everything in this
wave is **apparatus** — means, not end. It did not move the exact score and does not claim to.
The mission is a lower exact score; this wave makes MAIN less forgetful / more coherent so future
score-moving work loses less signal to compaction cliffs, drift-hook backstops, and context
pollution.

**STORES CONSULTED:** CLAUDE.md (subagent-contract / drift-detector / serializer / "off is a
tracked queue" non-negotiables); `docs/operating_manual_craft_handoff.md`; `tools/costate_digest.py`
(SessionStart path), `tools/triality_drift_detector.py` (leg heuristics), `tools/subagent_commit_serializer.py`;
`src/tac/subagent_contract.py` + its preflight gates + tests; `src/tac/jsonl_store.py`,
`src/tac/sidechannel_score_table.py` (locked/atomic-write patterns);
memory `all_arms_online_research_and_oss_authority_standing_20260720` (#767 verbatim);
`.omx/research/ddm_deferral_queue_ledger_20260729.md` (QA87/QA88, QD05); `~/.codex/config.toml`
+ `tools/codex_delegate.py` (QA87 audit). External anchor: OpenAI ARC-AGI-3 (Bigio & Sanders
2026-07-29) — retained reasoning + context compaction took GPT-5.6 Sol 13.3%→38.3% at 6× fewer
output tokens.

---

## Per-deliverable status (BUILT+TESTED vs DESIGNED)

| # | deliverable | status | evidence |
|---|---|---|---|
| 1 | MAIN hot-state manifest + SessionStart wire-in | **BUILT+TESTED** | `tools/main_hot_state.py` (seed/read/`--set-section`/`--json`, atomic fcntl-locked write); wired fail-open into `tools/costate_digest.py` main() after digest; seeded with verified live state (burn pid 68621, EMA-clamp GO pending, #782 boundary). Verified: session-start surfaces manifest, `--json` unperturbed, ruff clean. |
| 2 | contract retained-reasoning clause + #767 fold | **BUILT+TESTED** | `RETAINED_REASONING` + `RESEARCH_ORIGINAL_DESIGN_AUTHORITY` added to `tac.subagent_contract` (`__all__`, `CONTRACT_CONSTANT_NAMES`, `KEY_PHRASES`, composer, preflight required-constants). Both gates green (`check_subagent_contract_module_integrity`, `check_no_reasoning_echo_instructions`). 74 contract tests pass. |
| 3 | task-ledger archive custody manifest | **BUILT** | `.omx/research/ddm_hw1_task_archive_manifest_20260730.md` — 67 defensible SAFE-TO-ARCHIVE task#s + 60-row DO-NOT-ARCHIVE guardrail + the honest shared-number-space finding. MAIN applies deletions harness-side. |
| 4 | #713 recall-depth enforcement (warn-only) | **BUILT+TESTED** | recall-depth leg in `triality_drift_detector` (ledger/DAG append citing zero prior-artifact recall → ADVISORY, never blocks). 7 new tests. |
| 5 | shift-left leg classifier (serializer pre-commit advisory) | **BUILT+TESTED** | `owed_legs_line()` reuses the exact Stop-hook heuristics; serializer prints a fail-open pre-commit suggestion. 7 new tests. |
| 6 | QA87 codex settings audit | **BUILT** (enumeration; no codex runs) | table below |

---

## Deliverable 6 — QA87 codex-delegate settings audit (enumeration only, no runs)

**Model family match:** our delegate runs `gpt-5.6-sol` — the SAME family the ARC-AGI-3 receipt
measured (GPT-5.6 Sol). So the "harness settings are 3×-class levers" lesson applies directly.

**Config surface** (`~/.codex/config.toml` + `codex exec --help` + `tools/codex_delegate.py`):

| setting | value | class | note |
|---|---|---|---|
| `model` | `gpt-5.6-sol` | **SET-DERIVED** | operator-chosen |
| `model_reasoning_effort` | `xhigh` | **SET-DERIVED** | delegate sets `-c model_reasoning_effort=xhigh`. This is reasoning **EFFORT**, ≠ reasoning **RETENTION** across tool rounds |
| `approval_policy` | `never` | SET | autonomy |
| `sandbox_mode` | `danger-full-access` (per-invocation override to read-only/workspace-write in delegate) | SET | |
| `service_tier` | `fast` | SET | latency tier |
| context-window / **automatic-compaction** limits | **catalog default** | **DEFAULT-UNRACED** | config.toml comment: *"Context-window and automatic-compaction limits intentionally use the active model catalog defaults. Manual values here can silently truncate newer models."* — THE compaction-class lever the ARC-AGI-3 receipt names, left at default |
| reasoning **retention across tool rounds** | not CLI-exposed | **DEFAULT-UNRACED (not exposed)** | codex CLI exposes no retained-reasoning toggle; it is API/model-governed. Our application-layer analog IS the DAG/ledger/checkpoint/`_write_compact_prompts` resume discipline |
| session persistence (`--ephemeral`) | NOT set (sessions persist; `resume` works) | SET-by-omission | delegate relies on persisted sessions |
| history policy | catalog default | DEFAULT-UNRACED | |

**Crosswalk to the two ARC-AGI-3 levers:**
- **Retained reasoning** → EFFORT is SET (`xhigh`); RETENTION is not codex-CLI-exposed. Our
  cross-arm retained-reasoning analog is the apparatus this very wave hardens (hot-state manifest,
  contract retained-reasoning clause, DAG/ledger/checkpoint). So the lever is SET at the
  application layer, DEFAULT-UNRACED at the codex-internal layer.
- **Context compaction** → **DEFAULT-UNRACED** (catalog default, with an explicit truncation
  warning against manual override).

**NAMED RACE (future sol arm A/B; enumeration only — do NOT launch here):**
`race: codex-compaction-instrumentation`. Because manual compaction values risk silent truncation
on newer models, the honest first step is NOT to flip a knob blind but to **instrument**: log
per-arm context-window utilization + automatic-compaction events from `codex exec --json`, on
long multi-tool-round arms. THEN, if the catalog default is measured to truncate reasoning on our
long arms, A/B an exposed context/compaction override vs default, scored on arm success + output
tokens (the ARC-AGI-3 metric). Sister: our DAG/ledger/checkpoint discipline is already the
cross-death retained-reasoning + compaction analog; QA88 hardens it. Verdict scope: INSTANCE
(one config snapshot); a compaction-knob verdict needs the instrumented A/B.

---

## What is BUILT vs DESIGNED (honest)

- **BUILT+TESTED:** deliverables 1, 2, 4, 5 (code + passing tests + gates green + ruff `--select F`
  clean on every touched file).
- **BUILT (research artifact, no code):** deliverables 3 (manifest) + 6 (audit table). The archive
  DECISION (3) and the compaction A/B (6) stay with MAIN / a future arm — correctly, because the
  authority (harness task list; live arm dispatch) is not mine to actuate.
- **NOT done / out of scope:** no compaction knob was flipped (truncation risk); no codex runs
  launched; the burn (pid 68621) was untouched; no scorer/MLX/Metal compute.

## Self-review (own round 1, operating-manual §6)

- **Fail-open everywhere it must be:** the SessionStart wire-in, the shift-left advisory, and the
  recall-depth leg are each wrapped so a failure prints nothing / returns []/"" and never blocks a
  session or a commit. Verified: drift hook still exits 0; serializer `--help` still parses; digest
  `--json` unperturbed.
- **Class not instance:** the shift-left classifier + recall-depth leg REUSE the drift-detector's
  own heuristics (single source), so they can't drift from the Stop-hook they shadow.
- **Would the tests pass if the code were broken?** No — the recall-depth tests assert both the
  fire case (advisory present) AND the silence cases (recall token, non-ledger, subject token); the
  shift-left tests assert both owed and opt-out/chore silence.
- **Pre-existing lint corrected, scoped:** 3 pre-existing ruff nits in `costate_digest.py` + 1
  F541 in the serializer + 1 RUF022 in the contract module were pre-existing at HEAD (verified via
  `git show HEAD:`); fixed minimally (or per-file-ignored, matching sibling convention) because I
  touched those files. Test-1's stale magic-number (22) was already RED at HEAD from an earlier
  workflow-v2 landing; corrected to the live value.
