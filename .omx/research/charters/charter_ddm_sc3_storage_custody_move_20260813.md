# CHARTER — ddm_sc3_storage_custody_move (2026-08-13)

STORAGE-CUSTODY CERTIFY-AND-MOVE (cn5's fire-order 5, elevated by measured pressure: Vertigo `/Volumes/VertigoDataTier` at 94%, 129 GiB free vs APDataStore at 64%, 678 GiB free). The live campaign retains payloads at GB scale (re1 3.8 GiB · pz4r 28 GB referenced · re1x will retain rendered/argmax/pose fields) — headroom protects the ALWAYS-KEEP-THE-PAYLOAD P0. Scorer-free; no Modal; sr2 precedent (certify-and-MOVE, never delete).

**OPERATOR DOCTRINES BINDING:** "no naive or toy or generic basis ever" (structural manifests, no ad-hoc moves) + "as much as possible locally" (this arm IS local infrastructure).

## OPTIMAL FORM
Custody arm (OPTIMAL_FORM_NA: no mechanism raced; reference form = the certify-or-block rule from CLAUDE.md disk hygiene — machine-readable certificate BEFORE any byte moves). NEVER delete: MOVE with full manifest + symlink plan.

## THE WORK (per cn5's fire-order, consumer store `/Volumes/APDataStore/pact/`)
1. Identify INACTIVE families on Vertigo — terminal-arm retained stores whose arms are FINISHED and whose fire-orders do not name them as near-term consumer stores. EXCLUDE (hard): `regen2` (CERTIFY-BLOCKED per cn5) · all live solve/terminal directories · re1's `probability_object_race/ddm_re1_20260813` (re1x's LIVE consumer store) · pz4r's `direct_v6` retained records (its LC2-salvage component is pinned with a live fire trigger — verify before classifying; when in doubt, SKIP) · anything an active fire-order names.
2. Per family: write the COMPLETE manifest FIRST (paths, bytes, sha256 or tree hash, provenance/producing arm, destination, symlink plan, rebuildability rationale) → MOVE to `/Volumes/APDataStore/pact/<family>/` → verify destination hashes → leave symlink or pointer manifest at the source path → only then remove source bytes. Fail closed on any hash mismatch.
3. Target: recover ≥100 GiB on Vertigo without touching a single live consumer store. Report before/after `df` numbers.

## OUTPUT
`.omx/research/ddm_sc3_storage_custody_move_20260813.md` + per-family manifest JSONs alongside the moved data. Commit via `tools/subagent_commit_serializer.py` (post-edit shas, `[no-triality] [p0-ledger-ok]`). End with NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
