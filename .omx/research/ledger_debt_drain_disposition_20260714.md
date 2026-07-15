# Ledger debt-drain disposition — harness failure ledger + deferral ledger (2026-07-14, $0)

**Scope.** $0 debt-drain of `.omx/state/harness_failure_ledger.jsonl` + `.omx/state/deferral_ledger.md`.
NO-FAKE: nothing marked resolved/closed without a REAL verification (a passing check or the specific
fix commit/file). Pointer **UNMOVED 0.19108 / bank 0.18804** — hygiene + inventory only, no score claim.

**Headline correction to the task premise.** The prompt framed this as "36 unresolved rows, ALL
`failure_class='?'`". That is STALE. The harness ledger has 36 APPEND-ONLY *events* that fold into **11
failure states**, and the schema has no `failure_class` field at all — lifecycle is `surface` +
`resolution` (`open`/`worked-around`/`class-fixed`/`gate-landed`) + `causal_status`. 8 of 11 were already
resolved before this pass; the deferral ledger was likewise already drained (D4 hygiene 07-09 · burn-down
07-10 · reactivation-campaign FIRE-3 re-verified every trigger 07-10 · rate-law D36-D39 closed-measured
07-13). The debt was mostly already paid; this pass VERIFIES that and fixes one real data-integrity gap.

---

## A. Harness failure ledger — 11 states

Verified against current repo state at $0. Split: **7 resolved-verified · 1 worked-around (re-recorded
this pass) · 3 still-open-classified · 0 needs-investigation.**

| # | failure_id | surface | state | Verification (this pass) |
|---|---|---|---|---|
| 1 | `daemon_5min_harness_long_call_sweep_kill` | daemon | worked-around | RESOLVED-VERIFIED. Environmental (harness long-call sweep kills detached long-runners ~5min); 4 diagnoses falsified then MEASURED cause locked. Cure = bounded chunked foreground / codex-delegate harness. Correctly worked-around (not code-fixable). |
| 2 | `sigurg_144_harness_kills_bg_bash_process_group` | subagent | class-fixed | RESOLVED-VERIFIED. Detach pattern (nohup+subshell+disown+stdin-closed) documented in CLAUDE.md "Codex CLI invocation"; measured cause locked. |
| 3 | `serializer_whole_file_staging_absorbs_sibling_hunks` | tool | class-fixed | RESOLVED-VERIFIED. Guard `tools/check_sister_checkpoint_before_git_add.py` EXISTS (Catalog #314/#340). Residual (`--base-content-sha256` mandate) tracked as deferral D25. |
| 4 | `dashboard_hardcoded_gate_boundary_false_fail_at_init` | tool | class-fixed | RESOLVED-VERIFIED. Insufficient-data state + regime-guard landed (commits f35a0281e/600a07ed3). |
| 5 | `dashboard_false_FAIL_at_init` | (recurrence-chain) | class-fixed | RESOLVED-VERIFIED. Log→run-dir resolver sweep + bare-catch/null-guard JS sweep COMPLETE (2026-07-08 hardening; 53/53 dashboard suites). Row opened as recurrence-only (no `opened` event) but two `class-fixed` resolutions close both recurrences. |
| 6 | `launch_sh_inplace_rewrite_under_live_bash` | trainer-launch | gate-landed | RESOLVED-VERIFIED. `tools/launch_witness_run.py::write_launch_sh` writes ATOMICALLY (tmp + `os.replace`, fresh inode, L487-506) — the live-bash saved-offset corruption is structurally extinct. |
| 7 | `false_dead_diagnosis_incomplete_process_tree_walk` | tool | worked-around | RESOLVED-VERIFIED (worked-around). Discipline: liveness requires a `ps` process-TREE walk, never a grep pipeline. Residual gate tracked as deferral D26 (apparatus-batch). |
| 8 | `codex_workspace_write_sandbox_blocks_git_objects_20260712` | subagent | **worked-around (RE-RECORDED this pass)** | **DATA-INTEGRITY FIX.** The prior `mitigation_enforced` event carried `resolution="mitigated"` — NOT in the canonical `RESOLUTIONS` enum, so the lenient loader (`_validate` → skip) DROPPED it and the state mis-derived as fully-`open`. Re-recorded canonically as `worked-around`: harvest mechanism VERIFIED — `tools/codex_harvest_commit.py` + `tools/codex_delegate.py` exist (main harvests the sandboxed agent's on-disk build+test WIP and commits via the serializer). Underlying sandbox fs-isolation on `.git/objects` is an external-tool constraint → NOT class-fixed. |
| 9 | `zsh_nomatch_glob_aborts_monitor_scripts` | tool | **open (still-open-classified)** | STILL-OPEN. MEASURED environmental cause (harness shell is zsh; `nomatch` aborts on unmatched glob where bash leaves the literal — killed the v9 long-haul watch twice). No in-repo mitigation helper found (grep for `NULL_GLOB`/`unsetopt nomatch`/`2>/dev/null` discipline = none). Not code-fixable; not $0-reproducible-to-resolve. **Next-action:** a monitor/watch discipline — `setopt local_options null_glob` or quote/`2>/dev/null`-guard every bare glob in Monitor/loop scripts; candidate for the next apparatus-batch as a lint (sister of D26). |
| 10 | `codex_probe_token_limit_death_incomplete_wip_20260712` | subagent | **open (still-open-classified)** | STILL-OPEN. MEASURED: deep xhigh/ultra codex probes exhaust the per-agent token budget mid-build, leaving broken partial WIP on disk. External-tool constraint (not code-fixable). Partially covered by the codex-delegate harvest path (worked-around sibling #8) but the token-budget death itself is unmitigated. **Next-action:** scope codex probes below the ultra token wall (chunked/checkpointed builds) + the codex landing-review gate (memory `codex_landing_review_gate_two_landing_20260714`) surfaces the incomplete WIP. Owner: codex-delegate harness. |

**Genuinely-open harness debt that remains (2):** `zsh_nomatch_glob_aborts_monitor_scripts` (#9) +
`codex_probe_token_limit_death_incomplete_wip` (#10) — both external-shell/external-tool constraints, not
$0-code-fixable. Each has a named next-action above; neither blocks the score.

**Also observed (not fixed — informational):** 2 legacy pre-schema rows in the JSONL carry a non-`v1`
schema (`self_protected` + a bare disposition row, no `failure_id`) and are silently skipped by the
canonical lenient loader by design. They are harmless (never fold into a state) — left as-is per
APPEND-ONLY discipline; flagged so a future reader isn't surprised the raw line count (36) exceeds the
canonical event count (34 valid + 2 skipped).

---

## B. Deferral ledger — disposition

The ledger was already comprehensively drained. **Nothing is NEWLY-MET-and-$0** for me to close or route
beyond what is already routed. Per-bin disposition of the ~40 D-rows:

**Already CLOSED (verified this pass, not by me):** D3, D4, D20 (CLOSED-CONFIRMED 07-09) · D10 marimo
(RESOLVED-VERIFIED 07-10) · D36, D37, D38, D39 (rate-law CLOSED-MEASURED 07-13). = **8 closed.**

**Terminal-measured (correctly not reopened):** D41 (FORMULATION-DEAD-ON-HEADROOM — $0 reopen cert
2026-07-14) · D42 (MEASURED 2026-07-14, stays OPEN-CUSTODY: reopen = RE-CAPTURE of retained
logits/Jacobians, NOT a $0 recompute — store-nothing discipline). = correctly parked.

**Correctly GATED, trigger NOT-MET (verified $0 this pass):** D1, D2, D8, D9, D15, D16, D17, D18, D19,
D27b (chosen-chain GOVERNED STOP / FINAL terminal-band ckpt) · D5, D6 (#355 compute audit) · D7
(check-at-compile) · D21, D22 (post-launch byte-close of the chosen chain) · D23, D24, D26 (v8
gate) · D25 (session-quiesce) · D27 (machine-free) · D28 (config edit). I checked whether a FINAL
ckpt / governed stop now exists: `levelset_best.json` files DO exist (v9_cgauge_432, v752_baseline,
owed16v2, …) but the trigger is a governed *terminal-band* stop (`tools/witness_telemetry_audit.py
--section terminal_band → .d27b_ready`, muon-fired AND d_seg rel-slope < 5e-3), NOT the mere existence
of a best-ckpt. That certificate has not been declared. Firing these rows = running solves / A-Bs /
benches on boundary_math + witness_dsl surfaces = heavy + arm-owned = **scoped OUT → ROUTE, not execute.**

**OPEN, needs a build (not $0):** D40 (organ causal-OPE: log exploration in schedule-arm decisions —
trainer/causal-manifest change, owner `main`) · D43-D53 throughput reactivation queue (each requires a
real-n600 charged-cycle / Metal-wall receipt = NOT $0; the memory `no naive→binary no-go` principle is
already encoded per-row). = correctly OPEN with named receipts; route to their owners.

**BLOCKED:** D53 (transient task #495 identity unregistered — no technical conclusion possible; blocked-identity).

### One routing FINDING (not a close): dual-chain triggers now lag the live vehicle
D1/D2/D8/D15/D16/D17/D18/D19/D27b are pointed at the **"chosen-chain (v7.5.2|v8)"** dual-chain (07-09
D4-hygiene re-point). The live vehicle line has since moved to **V9·CGauge** (07-11..14; texture-trunk
DROPPED, bank 0.18804). The underlying condition (a governed terminal stop with a FINAL ckpt to solve on)
is vehicle-agnostic so the rows remain correctly ARMED-not-met — but the *label* is now one vehicle-gen
stale. **Routed to:** the next deferral-ledger hygiene pass / campaign owner — re-point "v7.5.2|v8" →
"V9·CGauge chosen chain". Not rewritten here: `deferral_ledger.md` is gitignored + main-only + absent
from this worktree, and the re-point is a campaign-curation judgment, not a $0 mechanical fix.

---

## C. Final disposition split

**Harness failure ledger (11 states):**
- resolved-verified: **7** (#1-7)
- worked-around, re-recorded this pass (data-integrity fix): **1** (#8)
- still-open-classified: **2** (#9 zsh-nomatch, #10 codex-token-limit — both external-tool, named next-actions)
- needs-investigation: **0**

**Deferral ledger:**
- closed (verified, already): **8** (D3/D4/D10/D20/D36/D37/D38/D39) · +2 terminal-measured parked (D41/D42)
- routed / correctly-gated-not-met: **~23** (dual-chain + spec + byte-close + machine-gated rows; ROUTE not execute — heavy/arm-owned)
- not-yet (open-needs-build, named receipt): **D40 + D43-D53** (real-n600 / Metal-wall receipts, not $0)
- blocked: **D53** (identity)
- 0 stale/broken triggers found; 1 routing finding (dual-chain label lags V9·CGauge — routed to next hygiene pass).

**Genuinely-open debt that remains + next-action:** (harness) zsh-nomatch lint + codex-token-budget
chunking, both apparatus-batch; (deferral) D40 causal-OPE logging build (owner `main`) + the throughput
D43-D53 receipts (each needs a real-n600 run, owner = costate/throughput feed). None is $0-closable and
none blocks the pointer.

**Pointer 0.19108 / bank 0.18804 UNMOVED — this pass moved no score; it corrected one dropped ledger row
and verified the rest of the debt was already paid.**
