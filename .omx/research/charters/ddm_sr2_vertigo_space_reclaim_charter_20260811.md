# ddm_sr2 — Vertigo certify-or-block space reclaim (MOVE, never delete; terminal headroom)

## Mission

/Volumes/VertigoDataTier is 99% full (24 Gi free) while /Volumes/APDataStore has 928 Gi. The
ps135 terminal (~08-12 late) needs Vertigo headroom for nothing — new stores already route to
APDataStore — but 24 Gi is dangerously thin for ANY surprise write. Reclaim space by
CERTIFY-AND-MOVE (the CLAUDE.md disk law: move + machine-readable record + symlink; destructive
delete FORBIDDEN except trivial caches).

## Ordered work

1. **SURVEY:** du -x top-level dirs under /Volumes/VertigoDataTier/pact (and any sibling roots);
   rank by size; classify each ≥5 GB dir: {LIVE (solve/terminal stores — ddm_ps135_20260810 +
   pr135_joint_solve_20260810 + terminal_watch = UNTOUCHABLE, exclude entirely) ·
   ALREADY-MIRRORED (in submittable_custody_mirror_20260811 — verify sha before acting) ·
   COLD (mtime > 14 days, no live consumer) · UNKNOWN (leave alone)}.
2. **CONSUMER CHECK per candidate:** grep .omx/research + .omx/state ledgers for path citations;
   a cited path may still MOVE if a symlink preserves it — verify the symlink resolves after.
3. **MOVE the certified set** to /Volumes/APDataStore/pact/vertigo_coldstore_20260811/<name>/
   via rsync -a + sha-256 tree manifest (before/after spot verification on the largest files) →
   THEN remove the source and leave a SYMLINK at the original path + a MOVED.json record
   {original_path, bytes, tree_sha_sample, destination, reason}. Stop when Vertigo free ≥ 150 Gi
   OR the certified-cold set is exhausted (report which).
4. **REPORT:** freed bytes · moved dirs table · anything ambiguous left alone (UNKNOWN list with
   why). Durable memo `.omx/research/ddm_sr2_vertigo_space_reclaim_20260811.md`.

## Boundaries

NEVER touch: ddm_ps135_20260810/** (LIVE SOLVE — pid 26406 writes here) ·
pr135_joint_solve_20260810/** · terminal_watch/** · anything mtime < 48h · the mirror itself on
APDataStore. NO deletes without a move-first (trivial .DS_Store/cache exempt). NO scorer, no
Modal, no git surgery. Verify each move's spot shas BEFORE removing source; on ANY mismatch,
keep both copies and flag. Serializer commit for the memo; BLOCKED-GIT ⇒ fire-order.

## OPTIMAL FORM

SCOPE = the full ≥5 GB population on the volume, each explicitly classified (report the
denominator — no silent skips, m50). PRIOR-LAW PREDICTION (derived from the volume being 1.8 Ti
at 99% with the active campaign only ~6 GB): ≥1 Ti of the usage is >14-day-old campaign-era bulk
(old run dirs, inflated-frame caches, VJP shards) of which ≥300 GB certifies as cold-movable in
this pass. FALSIFIER: if the cold set is <50 GB (i.e., the disk is genuinely full of live/recent
objects), report that honestly — the cure is then operator-level (bigger tier), not hygiene.
