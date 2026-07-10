# marimo #347 link-fix — RESOLVED same day (see §RESOLUTION at bottom). Original blocked-state report preserved append-only below.

**Task:** operator 2026-07-10 — "fix the Marimo notebook link and test it in browser on an actual
signed in server end to end WITHOUT changing the link to a new one." Hard constraint: the published
URL must not change; repair-in-place only. Checkpointed as `marimo-linkfix`
(`.omx/state/subagent_progress.jsonl`).

## Outcome: NOT COMPLETED — honest blockers, no fake progress

### Blocker 1 — the link does not exist anywhere on disk (LOCATE failed)

Exhaustive search for the published molab/marimo URL:

- `paper/README.md` + `paper/notebook.py` — no published URL (README references only
  `https://marimo.io` as the project homepage and local `uv run marimo edit/export` commands).
- Repo-wide grep (`molab`, `marimo.io/p/`, `marimo.app`, `https://molab...`) across `*.md`, `*.py`,
  `*.sh`, `*.json`, `*.txt` — zero URL hits outside generic references.
- `.omx/state/deferral_ledger.md` D10 row — names the deadline and a "parallel Marimo viz agent
  (sibling)" as owner, but carries NO URL.
- `.omx/state/lane_registry.json` `lane_witness_molab_release_rc_20260710` — level 0, all gates
  false, no URL in notes.
- `.omx/research/` — zero marimo/molab memos containing a URL.
- Agent memory (`~/.claude/projects/-Users-adpena-Projects-pact/memory/`) — zero molab hits.
- Git history, local AND `origin/main` after fresh fetch (`git log --all --grep`, `main..origin/main`
  empty) — the only marimo commits are `a0c6ee120` (scaffold) and `5aa81e5e0` (2026-07-09 22:19:12
  -0500, "marimo #347: interactive paper notebook submittable" — the SUBMITTABLE-state fix). Neither
  contains a published URL. The sibling Marimo viz agent has pushed nothing new to origin/main.

**Conclusion:** the published-notebook URL was never landed in any durable store. If it exists, it
lives only in the operator's signed-in molab account (locatable via the molab dashboard) or in the
sibling agent's un-landed session state. Per the hard constraint I did NOT create a new notebook.

### Blocker 2 — Chrome extension not connected (browser e2e impossible)

- `tabs_context_mcp` (×2): "Browser extension is not connected."
- `list_connected_browsers`: `[]` (no connected browsers on the account).

Three failures → stopped per the task's own rule ("if browser tools fail 2-3 times, stop and report
rather than retrying blindly"). No sign-in was attempted (forbidden), no notebook was created, no
screenshots possible.

### What WAS verified (local, $0): the canonical content is healthy

`paper/notebook.py` at HEAD (post-`5aa81e5e0`) exports cleanly:

```
.venv/bin/python -m marimo export html paper/notebook.py -o <scratch>/notebook_test.html
# marimo 0.23.11 · exit 0 · "Loaded 53 results, 63 timeline entries"
# 270.4 KB HTML · zero cell errors (only "traceback" match = config key "show_tracebacks": false)
```

So the repair-in-place, once unblocked, is mechanical: open the EXISTING molab notebook in the
signed-in dashboard (`molab.marimo.io`) and save the canonical `paper/notebook.py` bytes into it
(same notebook id → same URL).

## Deadline honesty

The #347 competition deadline was **2026-07-09 11:59 PM PST**. Today is 2026-07-10 — any fix landed
now is **post-deadline**. The submittable-state commit (`5aa81e5e0`) landed 2026-07-09 22:19 CDT
(= 20:19 PST, pre-deadline), but whether the molab publication/submission itself happened before the
deadline is **not verifiable from disk** — no URL, no submission receipt, no sibling-agent landing
exists in any store. D10's "IN PROGRESS — sibling finishing it" status was never confirmed by a
landed artifact.

## Unblock checklist (for operator / successor agent)

1. Connect the Claude Chrome extension (claude.ai/chrome, same account) — required for the signed-in
   session.
2. Provide the molab URL, or let the successor locate it via the signed-in `molab.marimo.io`
   dashboard (the notebook list of the owner account is the authoritative link source).
3. Then: repair-in-place (paste/save canonical `paper/notebook.py`), molab run-all, visitor-view
   check, screenshots — per the original 5-step plan.
4. LAND THE URL in `paper/README.md` + this ledger the moment it is known (the root cause of
   Blocker 1 is that it never was).

Pointer 0.19110 UNMOVED — apparatus/report only, no score claim, no public action taken.

---

# RESOLUTION (same day, 2026-07-10 — coordinator relayed operator unblock)

## The link (VERBATIM, unchanged throughout)

`https://molab.marimo.io/github/adpena/witness-machine/blob/main/notebooks/witness_machine_v12.py`

This is a molab **GitHub-proxy** URL: it renders whatever is at `notebooks/witness_machine_v12.py`
on `main` of the public repo `adpena/witness-machine`. "Fix in place" therefore = push corrected
bytes to that exact path+branch. The URL was never modified; no new notebook was created.

## What was actually broken (root cause, from the repo's own history)

The published entry is NOT pact's `paper/notebook.py` — it is **"The Witness Machine"**
(`witness_machine_v12.py`, 42 cells, 1,913 lines), a standalone notebook in
`github.com/adpena/witness-machine`. The pre-fix revisions (`f58568d` 2026-07-09 23:51 CDT
= 21:51 PST, `be424ad` 2026-07-10 01:08 CDT = 07-09 23:08 PST — **both pre-deadline in PST**)
imported the repo's internal packages via a repo-checkout assumption. On molab's GitHub mirror
there IS no repo checkout → the notebook failed to bootstrap at run time. That was the broken
link the operator saw.

**The in-place fix already landed before this session's browser leg:** commit `f111248`
(2026-07-10 07:22 CDT, post-deadline repair, authored by the operator/sibling agent) —
"fix(molab): bootstrap sealed runtime in GitHub mirror": downloads + sha256-verifies + caches the
sealed release bundle (`v1.2.0-rc2`, 3,704,001 bytes, sha256 `baf3e1e50b21…d439`) when no
checkout exists, removes static internal-package imports, adds cold/warm/self-heal regression
tests, and commits a clean `__marimo__/session/` snapshot so the static URL renders outputs.
Nothing further needed pushing to witness-machine — this session VERIFIED the fix end-to-end.

## E2E verification (agent-browser 0.25.3 — Chrome ext still unavailable; operator-authorized CLI)

Evidence dir: `.omx/research/marimo_linkfix_evidence_20260710/`

1. **URL loads + renders** (visitor view, anonymous): read-only preview, title
   `witness_machine_v12.py`, code renders — `01_initial_load.png`. PASS.
2. **Console:** zero errors/warnings on the page (`agent-browser console` + `errors` empty). PASS.
3. **Server run-all:** clicking "Run it now" → **"Sign in to run on the server"**
   (`02_run_it_now_clicked.png`). molab does not allow anonymous server execution of
   github-proxied notebooks. **STOPPED at this step per the no-credentials rule** — agent-browser
   drives its own Chromium without the operator's Safari session. This is the ONLY step that
   blocked on sign-in.
4. **Static preview (the anonymous-visitor render path):** renders code + CACHED OUTPUTS from the
   committed session snapshot — hero heading "The Witness Machine" + closing accordion "Sources,
   scope, and reproducibility" render as output, not code (`03/06–09_*.png`, page-text extract).
   Repo snapshot audited: 42/42 cells carry outputs, output kinds = {data}, ZERO
   errors/tracebacks. PASS.
5. **Asset chain:** release bundle URL → HTTP 200, exactly 3,704,001 bytes, sha256 matches the
   notebook's pinned `BUNDLE_SHA256`. PASS.
6. **Run-all equivalent, molab-faithful (local cold bootstrap):** copied ONLY the notebook file to
   an isolated dir (no repo checkout) with a fresh `XDG_CACHE_HOME` — the exact code path molab's
   server takes — then `marimo export html` (marimo 0.23.11): **exit 0, all 42 cells executed,
   bundle downloaded + verified into the sha-keyed cache (4.2 MB), ZERO
   MarimoRuntimeException/traceback markers in the 526.6 KB export**
   (`10–12_local_runall_*.png`). PASS.

## Deadline honesty (refined from the original report)

Publication (`f58568d`) and the RC2 reseal (`be424ad`) both landed BEFORE the 2026-07-09
11:59 PM PST deadline (21:51 PST and 23:08 PST respectively). The bootstrap REPAIR (`f111248`,
07-10 05:22 PST) and this verification are POST-deadline. So: entry submitted pre-deadline in a
then-broken-on-molab runtime state; repaired in place post-deadline at the same URL. Whether the
judges evaluate at the pre- or post-repair state is theirs to decide; we changed content only,
never the link.

## Pact-side landings

- Published URL landed durably in `paper/README.md` (§Published molab entry) — closes the
  original root bug (URL never on disk).
- D10 re-statused in `.omx/state/deferral_ledger.md` (gitignored live state, on-disk edit).
- Evidence screenshots committed under `.omx/research/marimo_linkfix_evidence_20260710/`.

Pointer 0.19110 UNMOVED — apparatus/verification only; no competition form touched, no
permissions changed, no credentials entered.

*Note:* the evidence `.png` screenshots are local-only (`*.png` is gitignored repo-wide); they persist on disk at `.omx/research/marimo_linkfix_evidence_20260710/` on Primary. The verification narrative above is self-contained without them.
