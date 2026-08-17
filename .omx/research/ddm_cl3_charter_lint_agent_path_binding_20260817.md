# ddm_cl3 — charter lint now binds to the Agent spawn path (task #1082)

**The three charter-lint legs now run on every Agent spawn, not only on codex
spawns.** I imported them into `tools/agent_model_routing_guard_hook.py` — the
only PreToolUse hook on matcher `Agent` — and landed a parity suite that fails if
the two spawn paths ever disagree on the same charter text. 28 new tests, 122
passing across the four affected suites, 0.02 s added per spawn.

**One finding the charter did not anticipate: the contract I was told to mirror
is not the contract the code runs.** `codex_arm_queue.cmd_add:1786` is
STRICT-BY-DEFAULT since 2026-08-13; the docstring at `:1712` still says
"Warn-only by default". Details and the measured blast radius are in §3.

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

I mirrored one contract detail that matters: **only `lint_charter_optimal_form`
can gate a spawn.** The two recall legs are advisory forever, exactly as
`cmd_add` treats them and as `codex_arm_queue:1213` states ("so
TAC_CHARTER_LINT_STRICT never upgrades an FM result into a refusal"). I also
mirrored the `recall_lint_na:` opt-out so the Agent path is never noisier than
the codex path on identical text.

## 3. The contract discrepancy — measured, not fixed

My charter said to mirror codex "EXACTLY: warn by default,
TAC_CHARTER_LINT_STRICT=1 blocks". That is what the `lint_charter_optimal_form`
docstring says at line 1712. It is not what the code does:

```
codex_arm_queue.py:1786   strict = os.environ.get("TAC_CHARTER_LINT_STRICT", "1") != "0"
```

Strict by default; the escape is `=0`. The docstring is stale relative to the
2026-08-13 "No naive or toy ever" flip, and the charter quoted the docstring.

I implemented warn-by-default as the charter instructed, because I measured what
the alternative would cost. I ran the optimal-form leg on **my own charter for
this task**:

```
optimal_form : ["build/race charter lacks '## OPTIMAL FORM' block ... and no
                OPTIMAL_FORM_NA:<rationale> waiver"]
```

Under strict-by-default the Agent path would have refused the very spawn that
commissioned this fix — and would refuse MAIN's ordinary working charters, on
the one hook that gates every Agent spawn. That is a blast radius I will not
take unilaterally.

Resulting agreement matrix. The **verdict** (the findings list) is identical in
all three columns; only the enforcement default differs:

| `TAC_CHARTER_LINT_STRICT` | codex path | Agent path | agree? |
|---|---|---|---|
| unset | REFUSE (rc=3) | WARN | **no** |
| `1` | REFUSE | BLOCK | yes |
| `0` | WARN | WARN | yes |

This is reported, not closed. Closing it means either flipping the Agent path to
strict (and conforming MAIN's charter template first) or flipping codex to warn.
Both are MAIN's call, not an arm's. Owner assigned in NEXT_IF_RESUMED.

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

I searched for a test file cited at `codex_arm_queue.py:1354` and got zero
results with `rc=0`. I nearly wrote "the cited test file does not exist" into
this memo. It exists — 6.9 KB, 11 tests, all passing.

The cause: `find` in this shell is not `/usr/bin/find`. It is a shell function
shim (`rtk`/`bfs`, from `~/.claude/shell-snapshots/`). Given a composite
expression it printed `rtk find: unknown flag '-print', ignored` and returned
**nothing, with rc=0** — indistinguishable from a genuine absence. `grep -rln`
found the file immediately.

This is the task's own genus reproduced in my instrument: a search that found
nothing and did not prove it looked. Reported, not fixed — the shim is operator
tooling outside my scope. The operating manual's rule held: re-derive from
primary artifacts rather than trust a summary, including a summary produced by
your own tool.

## 8. What I could not close

- **The enforcement-default divergence** (§3). Reported with the measurement;
  the decision is MAIN's.
- **The stale docstring** at `codex_arm_queue.py:1712`. It is the proximate cause
  of the wrong contract in my charter. I did not touch it — scope discipline says
  the codex path works and I am extending coverage, not editing it.
- **Advisory-leg scope.** `codex_arm_queue.lint_charter_recall_advisories` runs
  five legs; my charter named three. The Agent path runs the two shared recall
  legs plus the gating leg. `_lint_ownership`, `_lint_frontier_literals` and
  `_lint_bare_task_ids` remain codex-only. The parity suite compares like for
  like and does not paper over this.
- **No catalog number claimed.** Catalog #299's quota-under-400 gate is STRICT
  and the peek is 408. If this warrants a catalog row, MAIN claims it.

## NEXT_IF_RESUMED

| # | Item | Owner |
|---|---|---|
| 1 | Decide the enforcement default: flip the Agent path to strict (conform MAIN's charter template to carry `## OPTIMAL FORM` or `OPTIMAL_FORM_NA:` first), or flip codex to warn. Today the same charter is refused by one path and warned by the other. | MAIN |
| 2 | Correct the stale docstring at `codex_arm_queue.py:1712` to match `cmd_add:1786`. It caused the wrong contract in this task's charter and will cause the next one. | MAIN |
| 3 | Decide whether the three codex-only advisory legs (`_lint_ownership`, `_lint_frontier_literals`, `_lint_bare_task_ids`) should also bind to the Agent path. Bounded and mechanical if yes. | MAIN to route |
| 4 | Sweep the charters written during the codex wall (2026-08-15 → today) through the now-live lint and record what it would have caught. The gap's cost is currently unmeasured. | MAIN to route |
| 5 | Broken-search audit: `find` is shimmed and drops composite expressions silently. Any tool or arm that treats a zero-result `find` as proof of absence is exposed. | MAIN to route |
