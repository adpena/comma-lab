# Recovery session NO-SIGNAL-LOSS directive 2026-05-14

**Active recovery subagents** (in flight):
- `a155dcdabb5e1595d` RECOVERY-1 (D1 L2 fix + Catalog #220)
- `ae42f94dcb44b1d43` RECOVERY-2 (C6 finish + Modal harvest)
- `ade723fc2e7c5510e` RECOVERY-3 (per-trainer Tier B/C/D + XRAY Batch 6)

**Operator directive (verbatim 2026-05-14)**: *"be very careful to ensure no signal loss"*

This directive is **layered on top of all three recovery subagents' original prompts**. Every recovery subagent MUST treat the existing dirty tree + untracked files as PRECIOUS in-flight work from crashed sister subagents, NOT as orphan trash to clean up.

## NO-SIGNAL-LOSS PROTOCOL — MANDATORY for all 3 recovery subagents

### Rule 1: NEVER `git checkout --` / `git restore` / `git reset --hard` / `git stash` / `git clean` any tracked file

The dirty tree at recovery start contains ~30 modified files representing crashed sister subagents' in-flight work:
- `src/tac/substrates/d1_segnet_margin_polytope/__init__.py + archive.py + inflate.py` — RECOVERY-1's predecessor partial work
- `src/tac/substrates/c6_e4_mdl_ibps/tests/test_c6_substrate.py` — RECOVERY-2's predecessor C6 test edits
- `src/tac/substrates/vq_vae/*` — sister vq_vae subagent's work (NOT YOUR SCOPE; preserve verbatim)
- `experiments/train_substrate_*.py` (5+ files) — PER-TRAINER-WIRE-IN's partial Tier A wire-in (sane_hnerv + pr101 + balle + time_traveler + dp1 may be partially committed; pr101_lc_v2_clone enhanced_curriculum + vq_vae + a1_plus_lapose + d1 + c6 may be uncommitted)
- `src/tac/preflight.py` — multi-subagent shared file (Catalog #219 already landed; Catalog #220 number claimed; gate function in progress)
- `CLAUDE.md` — Catalog row edits in progress
- `src/tac/substrates/_shared/trainer_skeleton.py` — possible canonical-skeleton extension in progress

**Read before write. Inspect before commit. Use canonical serializer with `--expected-content-sha256` per Catalog #157+#216 ALWAYS — it will refuse if sister-subagent edits collide.**

### Rule 2: NEVER delete untracked `.omx/research/` files

Five untracked memos are codex/agent research artifacts from earlier today:
- `.omx/research/d1_z1_score_lowering_hardening_20260514_codex.md`
- `.omx/research/frame_exploit_cuda_transfer_audit_20260514_agent.md`
- `.omx/research/frame_exploit_segnet_posenet_20260514_pr106_mps600.md`
- `.omx/research/frame_exploit_segnet_posenet_20260514_pr106_mps64.md`
- `.omx/research/frame_exploit_selector_packet_20260514_codex.md`

These are HISTORICAL_PROVENANCE per Catalog #113. **Commit them via canonical serializer with a meaningful message; do NOT `rm`.**

### Rule 3: `.omx/state/subagent_progress.jsonl` is APPEND-ONLY

The crashed predecessors' checkpoint records are forensic evidence. **Read them, append your own, NEVER overwrite or truncate.** This is the canonical Catalog #131 + #128 + #206 protection.

### Rule 4: If a file you intend to modify is already in the dirty tree

1. **Read it first** to understand the predecessor's intent
2. **Inspect via `git diff <file>`** to see what changed
3. **Decide**: (a) commit predecessor's work verbatim then layer yours, OR (b) absorb predecessor's intent into your wave-coherent edit and commit both
4. **Use `--expected-content-sha256`** on the post-edit working-tree sha — Catalog #157+#216 will refuse if another sister subagent edits between your snapshot and the commit

### Rule 5: If a Modal call_id is in flight when you check, do NOT abandon it

Three Modal call_ids remain in the 24h cache:
- `fc-01KRKBF28G2M3N73FS7PDCB6AZ` (D1 R4)
- `fc-01KRKB7GFKQE8Y1JNKRYBWS3RJ` (D4 T4 smoke post-fix)
- `fc-01KRKA5DA13RH1CP5BQNDVAM3C` (D4 A10G; predecessor flagged as timed-out)

**Even if rc != 0, harvest the artifacts** — the trainer.log + run.log + provenance.json are diagnostic gold for what went wrong. Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE": never abandon a call_id without harvest.

### Rule 6: Sister-subagent file ownership IS SACRED

Each recovery subagent has a DISJOINT scope per the original prompt:
- RECOVERY-1: D1 substrate + Catalog #220 + CLAUDE.md non-negotiable section
- RECOVERY-2: C6 substrate + Modal harvest + C6 MDL ablation
- RECOVERY-3: per-trainer Tier B/C/D + XRAY Batch 6

If you need to modify a file you don't own, STOP and either:
- Coordinate via this directive file (append a note)
- Defer the edit to the owning subagent's next-cycle work
- DO NOT silently overwrite sister work

### Rule 7: Document EVERYTHING you skip or defer

If any pre-existing dirty state is "unclear ownership" or "appears orphan", do NOT delete it. Instead:
- Document in your memory file under "Preserved untouched (deferred to operator-route)"
- Surface in your operator-routable decisions list
- The operator decides what's signal vs noise; you don't

## What "signal" means in this context

Per CLAUDE.md "Long-burn score-lowering campaign default — NON-NEGOTIABLE": signal is anything that informs score-lowering, score-prediction, score-claim custody, dispatch decisions, math findings, anti-pattern fixes, or council deliberations. The crashed sister subagents were each producing signal toward score-lowering. Their partial work is signal.

**Signal-loss examples to AVOID**:
- Silently `git checkout --` a dirty file (loses predecessor's edits)
- Committing your work in a way that overwrites sister-subagent's parallel edits (Catalog #216 should refuse this; if it does, listen)
- Deleting an untracked `.omx/research/` memo because it "looks like scratch" (it's HISTORICAL_PROVENANCE)
- Clearing `.omx/state/subagent_progress.jsonl` history (it's the canonical Catalog #206 anchor)
- Re-running a Modal job whose call_id is still in the 24h cache (re-dispatch cost waste)

**Signal-preservation examples to PREFER**:
- Inspect-before-write
- Commit predecessor's verbatim work in a separate commit before layering your changes
- Append-only checkpoint discipline
- Read the directive files in `.omx/research/*_directive_*_20260514.md` for cross-subagent coordination
- Honor sister-subagent file ownership disjointness

## Cross-refs

- CLAUDE.md "Subagent coherence-by-default" non-negotiable (this directive uses the mandatory pre-read pattern)
- CLAUDE.md "Mandatory crash-resume protocol" (Catalog #206; the checkpoint store is the recovery anchor)
- CLAUDE.md "Subagent commits MUST use serializer" non-negotiable (Catalog #117+#157+#174+#216 protect against parallel-edit collisions)
- CLAUDE.md "Forbidden artifact-lifecycle violations" (Catalog #113; the 4-kind taxonomy LIVE_STATE/HISTORICAL_PROVENANCE/LIVE_RECIPE/DERIVED_OUTPUT)

Tagged `research_only=true`. NO score claims. NO GPU spend. Active recovery subagents pick this up on next checkpoint cycle or completion handoff.
