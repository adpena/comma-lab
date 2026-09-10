# ddm_mv1 — cold-storing an artifact with a `<path>.MOVED.json` manifest is the right owner behaviour, and it broke a sibling tonight: make every reader of a sibling's artifact RESOLVE the manifest (helper + the readers that matter + a preflight that refuses new bare `Path.stat()`/`open()` on cross-arm paths without it) (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, high) · Spawned by MAIN 2026-09-10. Sources: the incident — rp1 cold-stored its round-1 parse-back raw (`/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion/parseback/0.raw` → `/Volumes/APDataStore/pact/ddm_rp1_round1_bulk/parseback_0.raw`, manifest `parseback/0.raw.MOVED.json` with `moved_to`, `sha256`, `bytes`, `rebuildable_from`, `reason`); tc3's `public600` stage then died in `experiments/ddm_jg2_tail_reencode.py:238 file_fact` with FileNotFoundError on the old path (`/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/launch_public600/run.log`); CLAUDE.md "Local Disk, SSD Spill…" (certify-or-block; "leave a manifest or symlink when existing tools still need the original path" — APDataStore is ExFAT, so symlinks are not available and the manifest IS the mechanism); vr3/vr5/vr6 (the reclaim machinery must read MOVED manifests as certificates, not as absent files); "Bugs must be permanently fixed AND self-protected against" (two landings). Axes: apparatus; `score_claim=false`.

## MANDATE
(1) Helper `tac.artifact_moved.resolve(path) -> Path` (or the existing canonical location if one exists — grep for `MOVED.json` writers/readers first; rp1's writer is the reference schema; do not invent a second schema): if `path` is absent and `path.with_name(path.name + ".MOVED.json")` exists, verify the destination's bytes and (lazily, with a receipt) sha, and return the destination; chain up to a bounded depth; typed refusal on drift. (2) Wire it into the readers that consume sibling artifacts: `ddm_jg2_tail_reencode.py::file_fact`, the sj1/rp1/tc pricing surfaces' pin/`verify_pin` paths, `tac.candidate_seal` retained-payload checks, and `experiments/ddm_vr3_certified_raw_reclaim.py` (treat a MOVED manifest as a certificate row: the bytes live elsewhere; never "absent"). (3) Preflight (claim a catalog number; STRICT-eligible; same-line waiver): a `Path(...).stat()`/`open()`/`np.load` on a path under `/Volumes/*/pact/ddm_<other arm>/` in `experiments/ddm_*.py` without going through the helper is refused — warn-only at landing, strict-flip in the same batch if live count is 0 after wiring. (4) Writer side: a small `move_with_manifest(src, dst, reason, rebuildable_from)` that writes the manifest atomically AFTER the verified copy (rsync --checksum semantics; sha before/after), so owners have one sanctioned way to cold-store.

## PRIOR-LAW PREDICTION (m38)
- ≤ 6 reader sites need wiring; live count of bare cross-arm reads after wiring: 0–3 (each a one-line change); tc3's stage re-runs cleanly through the helper once the manifest resolves.
- **FALSIFIER:** if readers cannot be routed through one helper without changing the seal digest definitions, say so — the digests are measured objects (scg2); do not alter them.

## SCOPE
Apparatus only; ≤ 200 lines + tests; no SSD moves beyond a tmp fixture.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; no live-tree writes (rp1 round 2, tc3, vr6 are live). `.py` = 2 visible review passes + ruff; ≥ 12 tests (resolve / chain / drift refusal / writer atomicity / preflight positive+negative+waiver). Serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, `landing.patch` after ONE attempt. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_mv1`.
- Do not edit a live arm's running script's import surface without listing the live arms whose scripts you touch and landing those files LAST (the checkpoint-binding-drift law, 2026-09-10).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).

## PRIOR NEGATIVE SIGNAL
- pm2's landing broke sj1's checkpoints by editing a shared module mid-run — sequence behind live stages.
- vr4/vr5: "absent" is not a certificate — a MOVED manifest is.

## OPTIMAL FORM
- Reference form: pm2 (`8038f9e77`: fix + gate + tests in one landing) and the fire-tool sys.path fix (`6c74c56fd`). SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no symlink workaround (ExFAT); no second manifest schema.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Helper + writer + wired readers + preflight + tests + memo `.omx/research/ddm_mv1_moved_manifest_resolution_20260910.md` (live count per reader). Commit via the serializer. End with the live frontier line.
