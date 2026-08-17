# ddm_cl3 — charter lint now binds to the Agent spawn path (task #1082)

**The three charter-lint legs now run on every Agent spawn, not only on codex
spawns.** I imported them into `tools/agent_model_routing_guard_hook.py` — the
only PreToolUse hook on matcher `Agent` — and landed a parity suite that fails if
the two spawn paths ever disagree on the same charter text. 28 new tests, 122
passing across the four affected suites, 0.02 s added per spawn.

**One finding the charter did not anticipate: the contract I was told to mirror
is not the contract the code runs.** `codex_arm_queue.cmd_add:1786` is
STRICT-BY-DEFAULT since 2026-08-13; the docstring at `:1712` said the exact
opposite. Corrected in this landing (§3).

**Round 2 (MAIN adjudication, same day).** Three further fixes landed: the
inverted docstring is corrected with provenance; the hook now binds the UNION
entry points instead of three enumerated sub-legs (my own fix had reproduced the
allow-list genus one level down); and the strict-flip was measured and
**declined** — live count is not 0. Numbers in §3 and §9.

## 1. The defect, re-derived at source

Registry row `charter_lint_is_spawn_path_conditional_20260817` is confirmed. I
re-derived every line number in it; all four are exact, none moved:

| Symbol | Line | File |
|---|---:|---|
| `_falsified_premise_registries` | 1348 | `tools/codex_arm_queue.py` |
| `_lint_stale_numbers` | 1491 | `tools/codex_arm_queue.py` |
| `_lint_falsified_premises` | 1532 | `tools/codex_arm_queue.py` |
| `lint_charter_optimal_form` | 1702 | `tools/codex_arm_queue.py` |

I enumerated the hook matchers in `.claude/settings.json` directly. `PreToolUse`
carries exactly two matchers: `Bash` → `launch_guard_hook.py`, and `Agent` →
`agent_model_routing_guard_hook.py`. Codex arms spawn through Bash; Agent arms
spawn through the routing guard, which guarded model routing and nothing else.
The gap is real and the registry row states it correctly.

`decide(tool_input, env)` is at line 85 and its signature is unchanged.

## 2. What I changed

`tools/agent_model_routing_guard_hook.py`

- New `charter_lint(tool_input, env) -> (blocked, messages, status)`. It reads
  `tool_input["prompt"]`, stages it to an ephemeral temp file (the optimal-form
  leg takes a path), and calls the three legs.
- New `_load_lint_legs()` imports `codex_arm_queue` by path. The legs are
  imported, never copied. A second copy of a lint is a second thing to drift, so
  the parity suite asserts the hook source contains no `def` for any leg.
- `main()` calls the lint only after the model guard allows, inside its own
  `try/except`. Any failure allows the spawn and prints a loud stderr line.
- `decide()` is untouched and stays pure. The lint does file I/O, so it lives on
  its own surface rather than widening the pure decision function.

`src/tac/tests/test_agent_path_charter_lint_parity.py` — new, 28 tests.

`tools/codex_arm_queue.py` — docstring correction only, no behaviour change (§3).

I mirrored one contract detail that matters: **only `lint_charter_optimal_form`
can gate a spawn.** Everything else is advisory forever, exactly as `cmd_add`
treats it and as `codex_arm_queue:1213` states ("so TAC_CHARTER_LINT_STRICT never
upgrades an FM result into a refusal").

### 2a. Union entry points, not enumerated sub-legs (MAIN adjudication)

My first version named three sub-legs. MAIN was right that this reproduced the
task's own genus one level down: **an enumerated subset is an allow-list, and
allow-lists fail open.** A sixth leg landing inside
`lint_charter_recall_advisories` would have left the Agent path silently blind to
it — the exact defect I was commissioned to cure.

The hook now binds ENTRY POINTS:

| Entry point | Line | Bound | Why |
|---|---:|---|---|
| `lint_charter_optimal_form` | 1702 | yes — gating | the only leg that may refuse |
| `lint_charter_recall_advisories` | 1671 | yes — advisory | union of all 5 recall legs; owns `recall_lint_na:` |
| `lint_charter_capability_advisories` | 1251 | yes — advisory | cheap, deterministic, registry-read |
| `lint_charter_fm_advisories` | 1209 | **waived, with reason** | model-backed, `timeout=15` per call |

`lint_charter_fm_advisories` is the one genuine path-specific exclusion. It calls
`fm.charter_class(text, timeout=15)`. This hook is **synchronous in the spawn
path** on a measured 0.02 s budget; `cmd_add` is a queueing CLI where a 15 s
stall is tolerable and here it is not. That is a real asymmetry between the two
paths, not an oversight, and it is recorded in `_LINT_ENTRY_POINTS_WAIVED` rather
than left unmentioned.

Binding the union closed three legs that my first version left codex-only —
`_lint_ownership`, `_lint_frontier_literals`, `_lint_bare_task_ids`. Measured
proof, on the live hook:

```
$ echo '{"tool_name":"Agent","tool_input":{"model":"opus",
        "prompt":"Proceed with #1082 and #1079 as the basis."}}' | ...
[agent_model_routing_guard_hook] charter-lint WARN: RECALL: charter cites bare
task ids ['#1079', '#1082'] and NO memo filename — arms cannot resolve ...
```

`_lint_bare_task_ids` was unreachable from the Agent path an hour ago.

**The allow-list is now a deny-list.** `test_every_lint_entry_point_is_bound_or_
explicitly_waived` enumerates every `lint_charter_*` in `codex_arm_queue` and
fails if one is neither bound nor waived with a >40-char reason. Positive
control, run against a simulated sixth entry point:

```
unaccounted = ['lint_charter_brand_new_leg'] -> deny-list would FAIL: True
```

Latency after the rebind is unchanged at 0.02 s, 3 of 3 runs.

## 3. The docstring was an exact inversion — now corrected

My charter said to mirror codex "EXACTLY: warn by default,
TAC_CHARTER_LINT_STRICT=1 blocks". That is what the `lint_charter_optimal_form`
docstring said. It is the opposite of what the code does:

```
docstring :1713   "Warn-only by default; TAC_CHARTER_LINT_STRICT=1 refuses"
cmd_add   :1786   strict = os.environ.get("TAC_CHARTER_LINT_STRICT", "1") != "0"
```

The env var means the opposite of the headline. Strict is the default; `=0` is
the escape. This is the stale-headline genus
(`corrections_land_in_bodies_headlines_keep_the_stale_number_20260805`): the body
was corrected 2026-08-13 under the "No naive or toy ever" directive, the headline
kept the old semantics for four days, and then it propagated — MAIN read the
docstring, wrote my charter from it, and I implemented what was asked. **A
four-day-stale docstring produced divergent enforcement across two spawn paths.**

Corrected in this landing. The docstring now states strict-by-default, names
`cmd_add` as the authority for the env var's meaning, and carries the 2026-08-13
provenance so the next reader does not have to guess which of the two is current.
No behaviour changed in `codex_arm_queue.py`.

## 9. The strict-flip: measured, and DECLINED

MAIN's instruction was to match codex **only after driving live count to 0**, per
CLAUDE.md's Strict-flip atomicity rule. I measured. Live count is not 0, so I did
not flip.

Population: every charter file touched by git since 2026-08-15 (the codex wall).
Instrument: `lint_charter_optimal_form`, the only gating leg.

```
wall-era charter files on disk: 15
VIOLATORS (would be REFUSED under strict): 3 / 15
  - .omx/research/ddm_b2e_landing_and_charter_repin_20260816.md
  - .omx/research/ddm_rc2_regime_charter_and_lr_probe_20260816.md
  - .omx/research/ddm_cl3_charter_lint_agent_path_binding_20260817.md   [this memo]
```

Read honestly that is **2 genuine wall-era build charters**, plus one false
positive: this memo trips the placeholder-waiver check because it *quotes* the
string `OPTIMAL_FORM_NA:<rationale>` inside a lint message. Memos are not
charters and the live path never sees them, so that row is an artifact of my
measurement harness, not of the hook.

**But the decisive violator is not on disk at all.** MAIN's charters reach the
Agent tool as inline prompt text. I ran the gating leg on the charter that
commissioned this task:

```
optimal_form : ["build/race charter lacks '## OPTIMAL FORM' block ... and no
                OPTIMAL_FORM_NA:<rationale> waiver"]
```

MAIN's live charter template does not carry the block the CHARTER-TIME
OPTIMAL-FORM LAW requires. The lint is right to fire; the template is the defect.
Flipping strict today would refuse MAIN's next ordinary spawn on the one hook
that gates every Agent spawn.

MAIN's own criterion decides it: *"If there are other violators you cannot fix,
do NOT flip — report the count and leave strict opt-in with a named conformance
sweep."* I cannot fix MAIN's charter template from inside an arm. **Strict stays
opt-in on the Agent path.** The named conformance sweep is NEXT_IF_RESUMED #1.

Resulting agreement matrix. The **verdict** is identical in all three columns;
only the enforcement default differs, and that gap is now a tracked debt with an
owner rather than an undiscovered divergence:

| `TAC_CHARTER_LINT_STRICT` | codex path | Agent path | agree? |
|---|---|---|---|
| unset | REFUSE (rc=3) | WARN | **no — tracked, see #1** |
| `1` | REFUSE | BLOCK | yes |
| `0` | WARN | WARN | yes |

## 4. Proof it fires through the Agent path

Driven as the harness drives it — real payload on stdin, real process. Not
described; run.

**A. Advisory leg fires, does not block** (default env). Trigger string is a live
`claim_patterns` entry from the row registered today:

```
$ echo '{"tool_name":"Agent","tool_input":{"model":"opus",
        "prompt":"Proceed because charter recall is apparatus not volition."}}' \
  | .venv/bin/python tools/agent_model_routing_guard_hook.py
[agent_model_routing_guard_hook] charter-lint WARN: RECALL: charter restates
'charter recall is apparatus not volition' (?) — FALSIFIED premise. Origin
charter_recall_validation_is_apparatus_not_volition_20260816 ... verdict_scope=instance.
Re-derive before citing.
[rc=0]
```

**B. Advisory does not escalate under `STRICT=1`** — same text, `rc=0`, no deny
payload. The gating/advisory split is preserved.

**C. POSITIVE CONTROL — the lint can REFUSE through the Agent path:**

```
$ echo '{"tool_name":"Agent","tool_input":{"model":"opus",
        "prompt":"Build and measure a new codec arm."}}' \
  | TAC_CHARTER_LINT_STRICT=1 .venv/bin/python tools/agent_model_routing_guard_hook.py
[agent_model_routing_guard_hook] charter-lint REFUSED: build/race charter lacks
'## OPTIMAL FORM' block ...
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
 "permissionDecisionReason": "BLOCKED by tools/agent_model_routing_guard_hook.py
 charter lint (TAC_CHARTER_LINT_STRICT=1): ..."}, "decision": "block", "reason": "..."}
[rc=0]
```

**D. No prompt at all** — the charter asked me to state this explicitly. The hook
does **not** report "no findings". It returns `status="no-prompt"` and prints:

```
[agent_model_routing_guard_hook] charter-lint DID NOT RUN (status=no-prompt) —
this spawn was NOT examined; allowing UNLINTED (fail-open).
```

**E. Clean charter** — silent allow, `rc=0`, empty stdout and stderr. E and D are
distinguishable, which is the whole point.

## 5. Proof it fails open safely

The hook gates every Agent spawn, so this matters more than the coverage it adds.

- Lint raises → caught in `main()`, logged, loud stderr line, `rc=0`, spawn allowed.
- Legs unimportable → `status="unavailable:<type>: <msg>"`, `blocked=False`. Pinned
  by `test_lint_fails_open_and_loudly_when_legs_unavailable`, which forces an
  `ImportError` **with `STRICT=1` set** to prove strict mode cannot convert a lint
  bug into a refusal.
- One leg raises → that leg reports itself as unavailable; the others still run.
- Model-guard denial keeps its own reason; the lint never overwrites it.
- Every path returns `rc=0`. A PreToolUse hook must never exit non-zero.

Latency: 0.02 s per spawn, 5 of 5 runs (`/usr/bin/time -p`). The
`codex_arm_queue` import is 0.007 s and has no side effects at import.

## 6. The two clauses, implemented

**Clause 1 — allow-lists fail open.** I did not allow-list prompt shapes. The
hook lints whatever string prompt exists and announces loudly when there is
none. An unrecognised spawn shape is how this defect was born, so it is the one
case that must never be silent.

**Clause 2 — a search that finds nothing must prove it looked.** `charter_lint`
returns an explicit `status`, so "ran, 0 findings" and "never ran" are different
values, not the same empty list. Five parametrised tests pin that unlintable
inputs return `no-prompt`, and two positive controls assert the fixtures are
non-vacuous on **both** paths before any parity claim is made.

## 7. A live instance of the same genus, found during verification

I searched for a test file cited at `codex_arm_queue.py:1354`, got zero results,
and nearly wrote "the cited test file does not exist" into this memo. It exists —
6.9 KB, 11 tests, all passing. `grep -rln` found it immediately.

**MAIN asked me to restate this narrowly, and re-measuring narrows it further
than either of us had it. I ran the controls:**

| control | result |
|---|---|
| `find … -name "*falsified*"` | returns the file, `rc=0` |
| `find … -name "*falsified*" -print` | prints `rtk find: unknown flag '-print', ignored`, **then returns the file**, `rc=0` |
| same, `1>/dev/null` (stdout discarded) | **warning still appears** |
| same, `2>/dev/null` (stderr discarded) | **only the result appears** |
| my original `-prune` composite, ×3 | 368 matches, `rc=0`, deterministic |
| my original `-not -path` composite | returns the file, `rc=0` |

Two corrections follow, and one survives:

1. **My original claim is withdrawn.** I wrote that the shim "silently dropped
   the expression". It does not. Both composite forms I blamed return the file
   correctly now, and I **cannot reproduce** the zero-result. An unreproducible
   failure is not a finding.
2. **MAIN's narrowing is also not what the controls show.** The proposed
   restatement was that the warning goes to *stdout*, so a script redirecting
   results to a file gets the warning inside its data. Discarding stdout keeps
   the warning and discarding stderr removes it — the warning goes to **stderr**.
   A script redirecting results with `>` gets clean data. I am not restating it
   that way, because the measurement does not support it.
3. **What survives is smaller and is mine.** I ran the failing search with
   `2>/dev/null` and did not check `rc`. Whatever produced the empty output, I
   had thrown away both channels that could have told me the difference between
   "no matches" and "the command failed". The instrument discipline that follows
   is the one worth keeping: **on any search whose emptiness you intend to treat
   as evidence, do not discard stderr and do not ignore rc.** That is clause 2
   applied to my own hands, and it is the version I stand behind.

The operating manual's rule held twice here — re-derive from primary artifacts
rather than trust a summary, including a summary produced by your own tool, and
including one handed to you by the coordinator.

## 8. What I could not close

- **The enforcement-default divergence** (§9). Measured, declined, owned. Cannot
  close from inside an arm: it requires MAIN's charter template to conform first.
- **`lint_charter_fm_advisories` stays codex-only** (§2a) — a 15 s model call does
  not belong in a synchronous spawn hook. Recorded as a waiver with a reason, not
  left unmentioned.
- **Two pre-existing ruff errors** in `codex_arm_queue.py` (SIM300 at :1361,
  SIM905 at :1384). Verified present at HEAD before my edit by stashing; not
  introduced here and out of scope.
- **No catalog number claimed.** Catalog #299's quota-under-400 gate is STRICT
  and the peek is 408. If this warrants a catalog row, MAIN claims it.

## NEXT_IF_RESUMED

| # | Item | Owner |
|---|---|---|
| 1 | **Conformance sweep, then flip.** Add `## OPTIMAL FORM` (or a real `OPTIMAL_FORM_NA:` rationale) to MAIN's charter template, fix the 2 wall-era build charters, re-run the live count, and when it reads 0 flip `_LINT_STRICT_ENV` to default-on with `=0` as the escape — mirroring `cmd_add:1786` exactly. Blocked only on the template. | MAIN |
| 2 | Sweep the charters spawned during the wall (2026-08-15 → today) through the now-live union lint and record what it would have caught. Live count above measures the gating leg on 15 files; the advisory legs' would-have-caught set is still unmeasured. | MAIN to route |
| 3 | Audit the other `lint_charter_*` consumers for the same inverted-docstring class. One headline was 4 days stale and cost a divergent implementation; there is no reason to assume it is the only one. | MAIN to route |
| 4 | Decide whether `lint_charter_fm_advisories` should run on the Agent path asynchronously (post-spawn advisory) rather than not at all. Currently waived on latency grounds alone. | MAIN to route |
