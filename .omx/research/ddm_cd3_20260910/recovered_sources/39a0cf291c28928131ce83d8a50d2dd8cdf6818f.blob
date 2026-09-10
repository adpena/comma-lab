# COMMON ARM CONTRACT (every codex arm reads this)

Repo /Users/adpena/Projects/pact. Read CLAUDE.md + AGENTS.md +
docs/operating_manual_craft_handoff.md, then `.omx/state/main_hot_state.md` (live board).

## Sandbox note — READ THIS FIRST
You are spawned with `--add-dir /Volumes/VertigoDataTier/pact`, so the SSD cold-store tier IS
writable. A predecessor arm (fz3) DIED on `PermissionError: Operation not permitted` writing there
because that flag was missing. If you still hit a permission error on the SSD, STOP and report it —
do not silently symlink around it (that is how fz3 produced a custody caveat instead of a clean row).

## Binding constraints
- Commit via `.venv/bin/python tools/subagent_commit_serializer.py` with POST-EDIT
  `--expected-content-sha256` per file + `[no-triality] [p0-ledger-ok]`.
- NO co-author / Claude / AI trailer — commits are the operator's alone.
- `REVIEW_GATE_OVERRIDE=1` FORBIDDEN on `.py` (2 `tools/review_tracker.py` passes required; fine for .md/.sh).
- NEVER edit: `.omx/research/ddm_cr1_composition_row_827_20260801.md`,
  `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`,
  `src/tac/optimization/direct_description_carrier_compose.py`.
- Never touch the staged git index. No `git stash` on the shared worktree.
- No `/tmp` in persisted evidence — SSD tier for bulk, certify-or-block (record path, bytes,
  SHA-256, command/config before moving or deleting anything).
- MPS is NEVER score authority. GT decode ONLY via `frame_utils.yuv420_to_rgb`.
- `upstream/` is IMMUTABLE.
- ONE full-n600 scorer job at a time across the whole fleet, chunk ≤120. Your charter says whether
  you own the scorer slot. If you do NOT, do byte-only / scorer-free work and QUEUE the scorer step.
- Every number MEASURED with an axis label (`[macOS-CPU advisory]` etc.). Recompute S from
  components — evaluate.py's printed `Final score` is 2-decimal and LIES.
- verdict-scope every negative (INSTANCE / FORMULATION / FAMILY). n=8 banks NOTHING: a prefix of a
  skewed population is a different population, and the bias SIGN INVERTS between seg and pose axes.
  Use stratified-random n≥32 before banking anything.
- Follow-ons fire at harvest or they are orphan poison: every follow-on you name exits your run
  FIRED, FOLDED, or QUEUED-WITH-A-FIRE-ORDER. "Noted" is not a disposition.

## ORIGINAL RECALL — never charter-only (operator binding 2026-08-05)
Your charter's seed/context lists are a FLOOR, not a ceiling: they are MAIN's working memory once
removed and inherit MAIN's recall gaps. Before you adjudicate, price, compare, kill, or build
anything, do YOUR OWN recall against the FULL corpus — not just the codebase:
- `.omx/research/` memos + arm receipts (search by CONTENT for your surface, not only the names
  your charter handed you);
- the canonical equations registry (`.venv/bin/python tools/list_canonical_equations.py --json`);
- the research index / graph-memory surfaces (`.omx/research/CANONICAL_RESEARCH_INDEX*`, DAG
  `sub015_DAG_*` FEED blocks);
- design docs/SPECs + the task ledger rows touching your surface.
Your receipt MUST carry a `## RECALL EVIDENCE` section: sources searched (queries included), what
you found BEYOND the charter's seeds, and what it CHANGED in your plan (or "nothing found beyond
seeds in <scope>" — a scoped negative, never silence). "The charter didn't mention it" is never a
defense: the bd1 incident (a naive dense-field price reported as "the honest price" while
se3/rl1/sp1 already held cheaper measured representations of the same content) is the class this
clause extincts.

## Live frontier
S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600 (qo1 `sub_auto_pairbit`, sha
d5e814d5…). Gap to the PR130 bar 0.1721413 is 0.5818394 = seg 0.401519 (69.0%) / pose 0.0692658 /
rate 0.1110546 (uf1 re-derivation 08-05). Contest pointer 0.1910828242 is BORROWED and
harvest-only — never ours to ship.

## Final message
State what you MEASURED, what you did NOT, and every boundary. End with the own-vehicle frontier
line (S + bytes + axis; lead with a new number only if YOUR measurement moved it).
