# Serializer whole-file absorption — DIAGNOSIS + CLASS-FIX (`--base-content-sha256`), 2026-07-07

**Agent:** BUILD-WAVE-2 agent E (serializer class-fix). **Failure id:**
`serializer_whole_file_staging_absorbs_sibling_hunks` (harness-failure ledger, 5th+ firing).
**Fix commit:** `56fc64e19`. **Status: class-fixed** (ledger `resolution` row appended;
falsified/partial prior theories preserved in ledger history per append-only discipline).
Pointer 0.19110 UNMOVED (apparatus; `[no-triality]`).

## 1. DIAGNOSIS (measured, not assumed)

### 1.1 Reconstructed interleaving (serializer log × git show, all times 2026-07-07 UTC)

| time | event | evidence |
|---|---|---|
| <16:01 | dash-comb agent edits trainer (`--lane-band-dash-comb` wire-in) + `curriculum_dsl.py` (DashComb Lever) in the shared working tree, uncommitted | 9556d38f2 commit message + absorbed content below |
| 16:01:11 | telemetry agent commits **1d6704e5b** (trainer + perclass_verdict), `sha_present=True`, all checks pass | serializer log row; `git show 1d6704e5b -- <trainer>` contains the full `--lane-band-dash-comb` wire-in (argparse flag, `build_combed_lane_band_priors` import, `__cfg_lane_band_dash_comb` keys) — **absorption #1** |
| 16:06:31 | an agent tries to commit the trainer → `commit_failed` ("nothing to commit" class) — its hunks already landed under 1d6704e5b | serializer log row (head=c3f4a50a2, files=[trainer]) |
| 16:07:17 | GroundFrameChart agent commits **049aa0d9f** (`curriculum_dsl.py`), `sha_present=True`, passes | `git show 049aa0d9f` contains the `DashComb` Lever factory — **absorption #2** |
| 16:11:05 | dash-comb agent commits **9556d38f2**; trainer diff only 9 lines (residual hunks); documents the absorption in its commit message | git show --stat |

Content intact at HEAD in every case; the damage is **attribution + review-scope corruption**
(the absorbed hunks were "reviewed" and committed under bodies that never mention them), exactly
the class the ledger names.

### 1.2 Hypothesis verdict: **CONFIRMED**, with a sharpening

Hypothesis under test: *"the later committer's `--expected-content-sha256` is computed on the
already-merged working tree, so the check passes on content containing both agents' hunks; the
sha discipline only guards the lock-wait window."*

**CONFIRMED — and it is worse than 'the later committer':** *every* committer in the window had
`expected_content_sha256_present=True` and *every* check passed, because the documented
discipline (CLAUDE.md: "capture the working-tree sha of every file you plan to commit AFTER all
edits") computes the sha **on the merged file by construction**. Catalog #157 (rc=4, pre-lock
working-tree vs declared) and Catalog #216 (rc=5, staged blob vs declared) are **tautological
against content co-mingled before the caller's snapshot** — they only detect edits made *after*
it (the lock-wait / add windows). Temp-index staging (`temp_index=YES` on every row) is
irrelevant to this class: it isolates the *index* from concurrent staging, but `git add` reads
the shared *working tree*. The ledger's earlier framing ("per-hunk isolation is outside the
serializer's design") was correct about the gap; the missing *information* is the caller's edit
**BASE** — without it the serializer cannot distinguish "my edits on top of HEAD" from "my edits
on top of HEAD + someone's uncommitted hunks."

## 2. CLASS-FIX chosen: (a) `--base-content-sha256` — and why not (b)/(c)

**(a) landed** (commit `56fc64e19`): optional, repeatable
`--base-content-sha256 <relpath>=<sha256|new>` — the SHA-256 of the file's content **BEFORE the
caller's own edits** (`new` = caller-created file). The serializer compares the declared base
against the file's blob **at HEAD** (`git cat-file blob HEAD:<path>`), pre-lock (fast diagnostic)
and **again post-lock** (HEAD may move during the lock-wait):

- **base == HEAD blob** → the working-tree delta is exactly the caller's own edits →
  whole-file staging is attribution-safe.
- **base ≠ HEAD blob** → rc=6 REFUSE, loud: either foreign uncommitted hunks were present at
  edit-start (**absorption** imminent — the incident class), or HEAD moved past the base
  (whole-file staging would **REVERT** a sibling's landed hunks — a previously-uncaught reverse
  hazard the post-lock re-check now also closes).
- **Natural WAIT_AND_RETRY:** once the sibling lands exactly the hunks in the caller's base,
  `HEAD:<file>` equals the base and the retry passes — with correct attribution (proven by
  `test_base_sha_passes_after_sibling_lands`: the retry commit contains `+AGENT-HUNK` only).

**Why not (b) hunk-scoped staging:** reconstructing the caller's intended base→post diff needs
the base **content** (not a sha) plus git-apply machinery with its own failure modes (context
collisions on adjacent hunks); and when base == HEAD, whole-file staging already stages exactly
the caller's hunks — so (b) only adds value in the mismatch case, where silently splitting
authorship is the *wrong* default anyway (fail LOUD was the brief). (a) buys the same
attribution guarantee with ~1/10 the machinery.

**Why not (c) edit-intent registry:** it already exists — Catalog #340's sister-checkpoint guard
*is* an fcntl-JSONL edit-intent registry consulted by this serializer, and it demonstrably did
not fire in this incident (agents do not reliably register `files_touched` at edit start; 60-min
window + label-based self-exclusion make it porous). Building a second one duplicates a failed
mechanism ("never build a parallel registry").

**Fail-loud, simple, backward-compatible:** rc=6 (new, no collision with 2/3/4/5/8/9/10/11); no
behavior change without the flag; both refusal directions logged with forensic shas
(`base_content_sha_mismatch_pre_lock` / `_post_lock`, `declared_base` + `head_blob` per file);
`base_content_sha256_present`/`_file_count` stamped on every log row so forensics can
distinguish base-guarded from legacy commits when the class fires again.

## 3. Proof (scratch-repo concurrency tests — never the real repo)

`src/tac/tests/test_subagent_commit_serializer_base_sha.py` (10 tests, all in throwaway
`tmp_path` git repos with the module's `REPO_ROOT/LOCK_PATH/LOG_PATH` patched):

1. **Reproduction:** legacy flags (`--expected-content-sha256` on merged content) → rc=0 and
   `HEAD:shared.txt` contains `SIBLING-HUNK` — the absorption, reproduced.
2. **The fix:** same interleaving + declared base → **rc=6**, HEAD unmoved, forensic log row.
3. Positive control (base==HEAD) commits; 4. WAIT_AND_RETRY after sibling lands commits with
   clean attribution; 5–6. `new` token accept/refuse; 7. malformed flag rc=2; 8. post-lock
   re-check rc=6 on HEAD movement during lock-wait; 9. no-flag path byte-identical behavior;
   10. `_hash_head_blob_files` pins HEAD-blob (not working-tree) semantics.

Adversarial self-check: a broken fix (check unwired → rc=0) fails test 2; a wrong implementation
(compare vs working tree instead of HEAD blob) fails tests 3 and 10. Sister suites re-run green:
54 tests (serializer + concurrent-edit-detect + Check 117/157/174 gates) + 4-proc stress.

**Dogfood:** the fix's own landing commit `56fc64e19` ran through the new check live
(base=HEAD-sha for the serializer, `new` for the test file) and passed with exactly 2 files.

**Bonus hermeticity fix:** `_refresh_real_index_after_temp_commit` had a def-time
`repo_root=REPO_ROOT` default, so existing scratch-repo tests silently ran `git reset` against
the **real** repo's index; now resolved at call time.

## 4. Residual exposure (honest boundary)

The flag is **opt-in** — callers that omit it keep legacy absorption behavior (test 1 documents
this deliberately). Adoption path: the serializer docstring now carries the canonical 3-sha
discipline (BASE before editing, POST after, both flags). Follow-ons, not in this landing:
mandate base shas for hot shared files (extend the GAP-5 high-risk mandate — deferred because
a mid-flight mandate would refuse live siblings' in-flight commits today), and fold the 3-sha
discipline into `tac.subagent_contract` / CLAUDE.md serializer section. Multi-machine
coordination remains out of scope (fcntl is filesystem-local, per the serializer's standing
scope note).

Ledger: `record_diagnosis` (measured) + `record_resolution` (class-fixed) appended 2026-07-07;
prior worked-around rows and the earlier narrower theories remain in history.
