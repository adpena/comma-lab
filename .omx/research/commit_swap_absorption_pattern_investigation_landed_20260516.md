# Commit-swap absorption-pattern investigation + Catalog #314 self-protection landed 2026-05-16

**Lane**: `lane_commit_swap_absorption_pattern_investigation_20260516`
**Subagent**: `commit_swap_absorption_investigation_20260516`
**Anchor commits (bug class empirical receipts, today)**:
- `89d89c27e` "Harden L5 sideinfo and dispatch probe gates" (2026-05-16 22:04:51 -05 = 2026-05-17T03:04:51 UTC)
- `c09c6e1c8` "Preserve probe outcomes landing memo" (2026-05-16 22:06:27 -05)
- `5562afc3c` "Preserve L5 v2 TT5L audit signal" (2026-05-16 22:12:10 -05)
**Anchor commit (yesterday's WAVE-D)**: `2c957c31e` (2026-05-15)

## Root cause (premise verification per Catalog #229)

The operator's `/commit` slash command lives at `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit.md` (`commit-commands` plugin, enabled in the operator's global Claude settings). The command body literally instructs the LLM:

> "Based on the above changes, create a single git commit. ... Stage and create the commit using a single message."

It does bare `git add` + `git commit` directly (NOT through `tools/subagent_commit_serializer.py`). When invoked while a sister subagent has uncommitted edits in the shared working tree, the bare `git add` packages whatever the LLM thinks is relevant — which can include the sister's still-in-flight edits.

The resulting commit carries a terse LLM-generated subject like "Preserve probe outcomes landing memo" or "Harden L5 sideinfo and dispatch probe gates" and the sister's work is silently attributed to the wrong commit body. Catalog #157 `--expected-content-sha256` does NOT fire rc=4 because:

1. Sister subagent's serializer is called AFTER the absorbing commit lands.
2. At the moment of the serializer call, the working-tree content of the sister's declared files matches the sister's post-edit sha (the absorbing commit packaged the same bytes; the working tree IS that content).
3. The sister's `git add` of those files inside the temp index returns "no changes" because HEAD already has them.
4. The serializer commit succeeds with rc=0 and lands only the NEW files (e.g. the sister's landing memo) — the rest of the sister's work was attributed to the absorbing commit's body.

## Empirical signature (machine-readable detection)

Confirmed today via `.omx/state/subagent_progress.jsonl` + git log timestamp correlation:

| Bare commit | Time (UTC) | Absorbed in-flight subagent | Sister checkpoint time (UTC) | Δt (min) | Overlapping files |
|---|---|---|---|---|---|
| `89d89c27e` | 03:04:51 | `stc_v2_driver_fix_20260516` | 02:56:05 | 9 | `src/tac/preflight.py`, `CLAUDE.md`, `scripts/remote_lane_substrate_stc_v2.sh` |
| `89d89c27e` | 03:04:51 | `probe_outcomes_bake_in_20260516` | 02:54:57 | 10 | `src/tac/preflight.py` |
| `c09c6e1c8` | 03:06:27 | `stc_v2_driver_fix_20260516` | 02:56:05 | 10 | `src/tac/tests/test_check_152_modal_mounted_input_extension.py` |
| `1a2d84b3d` | 02:48:19 | `z6_phase_2_sextet_council_20260516` | 02:41:57 | 6 | `.omx/research/sextet_council_z6_phase_2_consensus_20260516.md` |
| `d26a07c90` | 02:40:14 | `z6_lift_subagent_20260516` | 02:27:28 | 13 | `.omx/operator_authorize_recipes/...` |
| `db035a7a6` | 02:29:10 | `1778984017_phase1brudin` | 02:13:37 | 16 | `experiments/train_substrate_rudin_floor_interpretable_ml.py` |

9 absorption-signature violations across 6 unique bare commits in the last 50.

## Six hypotheses, ranked by evidence (echoes WAVE-D 2c957c31e analysis)

| # | Hypothesis | Score | Verdict |
|---|---|---|---|
| H1 | Sister forgot to use serializer | 0 | REFUTED — sister DID use serializer; its commit (`7dd8a5412`) is in the log with all 7 expected_content_sha256 files. The race happened in the OPPOSITE direction: the bare commit fired BEFORE the sister's serializer call. |
| H2 | Sister bypassed Catalog #157 / #174 / #289 | 0 | REFUTED — sister passed `--expected-content-sha256` for all 7 files; OMNIBUS GAP-5 high-risk file gate (#289) was active. |
| H3 | Catalog #117 (`check_subagent_commit_serializer_uses_lock`) misfired | 0 | REFUTED — #117 IS detecting the bare commits (12 violations in last 50 today, including all 3 absorption commits) but is wired warn-only per legacy backlog. It DETECTS post-hoc; it does NOT BLOCK. |
| H4 | Catalog #157 missed pre-pre-lock race | 0 | REFUTED — at the moment of the sister serializer's lock-acquire, the working-tree content sha matched the declared sha because the absorbing commit had already packaged those same bytes. The pre-pre-lock check sees stable content. |
| H5 | **Bare-commit-from-`/commit`-slash-command absorbed sister edits while sister was still in-progress** | **10** | **DOMINANT** — empirically confirmed by 6 distinct (commit, subagent) pairs today. The pattern is structural: `/commit` does `git add` of the entire dirty working tree (or LLM-curated subset) and packages whatever it finds. |
| H6 | Cross-machine git race | 0 | REFUTED — all commits on same machine + same git repo; fcntl-lock is irrelevant (lock was never contested). |

## Fix landed (Option C per the prompt's design enumeration)

### Layer 1: STRICT preflight gate `check_no_subagent_files_touched_absorption_in_bare_commits`

Catalog #314 (~310 LOC including helpers). The gate:
1. Loads serializer commit hashes from `.omx/state/commit-serializer.log`.
2. Loads in-flight subagent rows from `.omx/state/subagent_progress.jsonl` (all rows, not just latest per subagent, because the absorption window matches against the timestamp closest BEFORE the bare commit).
3. Walks `git log -50 --format=%H%x09%cI%x09%s%x1e%b%x1f` (with `\x1e`/`\x1f` separators to robustly handle body newlines).
4. For each commit: if hash NOT in serializer log AND no waiver in body, parse file list via `git show --name-only --format= <sha>` (per-commit; clean separation).
5. Intersect commit's files with each in-flight subagent's `files_touched` (after exempt-file filtering); flag if overlap AND subagent's checkpoint is within `[commit_ts - 60min, commit_ts]`.

The detection is POST-HOC (the commit already landed). This is by design: we cannot block `/commit` from within preflight because preflight runs at commit time, not at edit time. The structural value is operator-facing alert + machine-readable evidence so absorption is detected within the same session.

### Layer 2: CLAUDE.md "Meta-bug class catalog" row 314

Full canonical row in CLAUDE.md documenting bug class anchor, root cause, hypotheses-ranked, sister gates, acceptance cascade, exempt files, test coverage, strict-flip cadence.

### Layer 3: 38 dedicated tests + live-repo regression guard

`src/tac/tests/test_check_314_subagent_files_touched_absorption_in_bare_commits.py` covers:
- Helper unit tests (ISO-UTC parser; waiver token detection with placeholder rejection; exempt-files set; waiver-tokens tuple).
- Loader unit tests (serializer hash loader; malformed-JSON tolerance; missing-file fail-OPEN; in-flight loader with exempt filtering; space-separated legacy format; multiple rows per subagent).
- End-to-end (clean repo; canonical absorption pattern detected; serializer-log commit skipped; absorption-pattern waiver accepted; no-serializer waiver rejected; placeholder waiver rejected; commits outside 60-min window skipped; commits before checkpoint skipped; no-file-overlap skipped; exempt-files-overlap skipped; dedup per commit-subagent pair; multi-subagent collisions counted separately).
- Strict-mode (raises with Catalog #314 message / silent on clean).
- Orchestrator wire-in warn-only regression guard + Catalog #185 sister-gate-callable regression guard.
- Anchor-specific live-repo regression guard pinning today's STC v2 + probe_outcomes + L5 + Z6 absorption pattern empirical effectiveness at landing.

All 38 tests pass.

## What I did NOT land (and why)

- **Option A (disable auto-commit during multi-subagent runs)**: cannot control the `/commit` slash command from preflight; only operator-side discipline applies. Out of mutation frontier (the slash command lives in `~/.claude/plugins/...`).
- **Option B (canonical serializer detects absorption AFTER commit)**: would require the serializer to also fail when its `git add` returns "no changes" while expected-content-sha256 matches. Cleaner future enhancement; today the bare commit is OUTSIDE the serializer so this layer cannot fire at the right time.
- **Option D (file-lock at edit-start)**: too heavy refactor; would require every subagent to declare file-intent at start and would block normal operator work. Operator-routable as a follow-up.
- **Option E (operator discipline only)**: already CLAUDE.md non-negotiable + this is the SECOND occurrence of the bug class. Documented warnings alone don't extinct the pattern.

## Live count + sister-gate cleanliness

- Catalog #314 (NEW): **9 violations** at initial landing under the broader checkpoint-status scan; **8 violations** after follow-up review tightened scope to status=in_progress only (warn-only). Strict-flip pending operator-routed audit + waiver backfill OR an absorption-clean window.
- Catalog #117 (existing sister): 12 violations at landing (warn-only; this gate's coverage is bigger than #314 since #117 catches ALL bare commits, not just absorbing ones).
- Catalog #118 (no duplicate numbers): **0 violations**
- Catalog #159 (catalog text matches preflight strict value): **0 violations**
- Catalog #176 (strict callsites have CLAUDE.md row): **0 violations** (#314 wire-in is warn-only so #176 does NOT require a row, but row added anyway for completeness)
- Catalog #185 (LIVE_COUNT drift detection): **0 violations**
- Catalog #186 (catalog claim committed via serializer): **0 violations** — claim of #314 went through `tools/claim_catalog_number.py --commit-via-serializer` per the standard pattern.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A — this is a structural preflight extension for commit-discipline, not a substrate sensitivity primitive.
2. **Pareto constraint**: ACTIVE — extending the commit-swap protection family strengthens the multi-subagent edit/commit coordination constraint; future absorption events are surfaced structurally rather than discovered weeks later.
3. **Bit-allocator hook**: N/A — no per-tensor importance change.
4. **Cathedral autopilot dispatch hook**: N/A — this gate does not affect dispatch ranking.
5. **Continual-learning posterior update**: N/A — no empirical anchor produced (this is process hardening; the next subagent commit wave produces the next anchor).
6. **Probe-disambiguator**: N/A — the bug class has a single dominant interpretation (H5); no 2+ defensible interpretations remain.

## Operator-action-required summary

1. **Operator-routable retro-waiver backfill on 6 anchor commits**: add `# ABSORPTION_PATTERN_OK:<rationale>` via `git notes append` to each of `89d89c27e`, `c09c6e1c8`, `5562afc3c`, `1a2d84b3d`, `d26a07c90`, `db035a7a6` with the rationale that today's session was unaware of the bug class structurally; once backfilled, Catalog #314 live count drops to 0 and strict-flip becomes atomic per CLAUDE.md "Strict-flip atomicity rule".
2. **Update the `/commit` slash command markdown** (in `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit.md`): add an instruction to FIRST query `.omx/state/subagent_progress.jsonl` for in-flight subagents AND if any rows are in-flight with `status=in_progress` whose `files_touched` overlap with the working-tree dirty set, REFUSE to commit and route the operator to `tools/subagent_commit_serializer.py` instead. This is the runtime protection that complements the post-hoc gate. Out-of-repo, so operator-driven.
3. **Sister WAVE-3-MODAL-TRAIN-LANE-FIX coordination**: the prompt mentions this sister may also touch `src/tac/preflight.py`. My Catalog #314 insert is at the END of preflight.py (after Catalog #313's last function, before `if __name__ == "__main__":`); the wire-in is between Catalog #303 and the start of the orchestrator's tail. WAVE-3 should be able to land non-overlapping edits.
4. **Sister-subagent ownership map per Catalog #230 honored**: my commit body explicitly cites this gate's ownership boundary (preflight.py + new test file + CLAUDE.md row + landing memo); sister WAVE-3-MODAL-TRAIN-LANE-FIX owns `experiments/modal_train_lane.py` + `src/tac/deploy/modal/mount_manifest.py` + possibly preflight.py extension at a different region.

## Lane gates marked

- `impl_complete=True` (Catalog #314 gate function + 38 tests + CLAUDE.md row + orchestrator wire-in all landed)
- `strict_preflight=True` (new STRICT preflight gate is the work itself)
- `memory_entry=True` (this memo)
- `three_clean_review=False` (single subagent landing; no recursive review cycle needed for warn-only)
- `contest_cuda=False` / `contest_cpu=False` (not score-affecting)
- `deploy_runbook=False` (preflight-only landing)

## Premise verification per Catalog #229

8 PVs confirmed PRE-edit:
1. `auto_commit.sh` exists and is the OPERATOR-side housekeeping helper (not the source of today's "Preserve X" / "Harden Y" commits — those have a DIFFERENT message style)
2. `/commit` slash command lives at `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit.md` and does bare `git add` + `git commit`
3. `89d89c27e` is NOT in `.omx/state/commit-serializer.log` (`Bash grep` confirmed)
4. `89d89c27e` includes `src/tac/preflight.py` (+586) + `CLAUDE.md` (+6) + 3 driver scripts that STC v2 FIX subagent declared in its checkpoint
5. `stc_v2_driver_fix_20260516` checkpoint at 02:56:05 declared `files_touched=['scripts/remote_lane_substrate_stc_v2.sh', 'src/tac/preflight.py', 'CLAUDE.md']`
6. STC v2 FIX's actual serializer commit `7dd8a5412` only landed the landing memo (135 LOC) — proving the other 6 files were already absorbed
7. Catalog #117 IS firing on these commits but wired warn-only (12 violations live)
8. Catalog #157 / #174 / #289 family don't apply because the absorbing commit doesn't go through the serializer at all

## Checkpoint discipline per Catalog #206

4 checkpoints written to `.omx/state/subagent_progress.jsonl`:
- step 1 (start: pre-flight reads done)
- step 2 (root cause confirmed; designing fix)
- step 3 (gate implementation landed; tests passing)
- step 4 (final: commit pending via serializer)

## Cross-references

- **Sister of**: Catalog #117 (subagent commit serializer must be used; broader scope than #314) + Catalog #157 (commit-swap pre-pre-lock hash) + Catalog #174 (--expected-content-sha256 mandatory) + Catalog #216 (post-stage hash) + Catalog #289 (drop-flag-and-retry pattern) + Catalog #230 (bulk-rewrite ownership map) + Catalog #248 (residual conflict markers) + Catalog #302 (sister subagent scope overlap via checkpoint JSONL).
- **WAVE-D forensic anchor**: `.omx/research/commit_swap_incident_2c957c31e_forensic_analysis_20260515.md` (yesterday's first observation; produced Catalog #289)
- **Today's surfacing anchor**: `.omx/research/stc_v2_driver_path_layer_fix_landed_20260516.md` (the STC v2 FIX landing where this bug class recurred and was surfaced via the cross-stack coordination notes)
- **Canonical bug-class incident memo**: `feedback_concurrent_subagent_commit_message_swap_20260429.md` (the 92aba3ca anchor that birthed Catalog #157)
- **CLAUDE.md non-negotiables honored**: "Subagent commits MUST use serializer" + "Bugs must be permanently fixed AND self-protected against" + "Subagent coherence-by-default" + Catalog #229 premise verification + Catalog #206 checkpoint discipline

---

Lane: `lane_commit_swap_absorption_pattern_investigation_20260516`
Subagent: `commit_swap_absorption_investigation_20260516`
Cost: $0 (pure preflight gate landing + tests + CLAUDE.md + landing memo)
Time: ~45 minutes
Commit: via canonical serializer with `--expected-content-sha256` per Catalog #157 / #174 / #289

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
