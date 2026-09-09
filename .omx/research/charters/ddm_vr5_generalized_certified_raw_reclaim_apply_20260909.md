# ddm_vr5 — turn vr4's 20 RETAINED_REPRODUCER rows (68.218 GiB of inflated `0.raw` from closed advisory arms) into a certified DELETE plan the vr3 executor can apply, generalized past vr3's two hardwired families; MAIN runs the apply outside the sandbox with the live process gate (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, high) · Spawned by MAIN 2026-09-09. Sources: vr4 (`.omx/research/ddm_vr4_both_ssds_certified_reclaim_successor_20260909.md`, `216993f0d`; ledger `.omx/research/ddm_vr4_reclaim_ledger_20260909.jsonl`: 20 rows with `certificate_status = RETAINED_REPRODUCER_VERIFIED_RAW_REHASH_AND_SAFETY_OWED`, all `0.raw` inflated frames of closed advisory arms — bz2d, wd2, hv1, sa3, ck1, ck2, jg4, jg5, cd1, fs3, ri1, ni1, rf1 on APDataStore; jf2, rr8, ni1 on Vertigo), vr3 (`experiments/ddm_vr3_certified_raw_reclaim.py`, `ff1095d49`: the plan/apply executor that deleted 18 raws with PRE_DELETE/DELETED journal rows, lsof + reference gates, reproducer revalidation — but ADMITS ONLY the AP1 and JF2 families by constant), MAIN's receipts: rehash `/Volumes/VertigoDataTier/pact/ddm_vr4_reclaim_20260909/main_rehash_20260909.jsonl` (one JSON row per path: path, sha256, bytes, present — being produced now; wait artifact-bound ≤ 780 s per check until it has 20 rows), process gate: MAIN ran `pgrep -fl` per owning arm and `lsof` on all 20 paths at 2026-09-09 ~23:50Z — no live process, no open handle (record this as MAIN's receipt in your memo with the timestamp; the apply re-runs it live). Axes: bytes `[measured]`; `score_claim=false`.

## MANDATE
Generalize vr3's planner so a row is admissible when (a) its reproducer chain passed vr4's check (archive present with recorded sha, runtime tree present, provenance carries an exact `sys_argv` reproducer, eval receipt present), (b) MAIN's rehash sha matches vr4's `historical_sha256` where one exists and is recorded as the current sha where none did, (c) the path is not under any live arm tree, live seal, or pointer row, (d) the owning arm is closed (memo present with a verdict). Emit `ddm_vr5_reclaim_plan_20260909.jsonl` (schema `ddm_vr3.reclaim_ledger.v1`-compatible, with a `family` field per owning arm) and the EXACT apply command (`experiments/ddm_vr3_certified_raw_reclaim.py apply --ledger … --expected-ledger-sha256 … --journal … --target-bytes …`) for MAIN to run. Do NOT run apply yourself (your sandbox cannot see processes, as vr4 found — the gate would be vacuous).

## PRIOR-LAW PREDICTION (m38)
- All 20 rows admissible; 68.218 GiB deletable; per-drive: APDataStore ≈ 51 GiB (15 rows), Vertigo ≈ 17 GiB (5 rows).
- **FALSIFIER:** any row whose rehash disagrees with its historical sha, or whose reproducer archive sha drifted, is BLOCKED with the reason — never "close enough". If < 10 rows admit, say which gate refused most and why.

## SCOPE
Planner generalization + plan ledger + apply command + memo. Apply is MAIN's. No other files touched on either SSD.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY. Never delete, move or rewrite any SSD byte in this charter. Never touch live arm trees (`ddm_sj1_*`, `ddm_rp1_*`, `ddm_bnd1_*`, `ddm_gb2_*`, `ddm_cmp2_compose`), seals, or pointer rows.
- Keep vr3's executor semantics (revalidate certificate + lsof + reference gates at apply; PRE_DELETE/DELETED journal; fsync; ledger updated atomically). Generalize by DATA (family table from vr4's ledger) not by adding more constants; keep the two original families working (tests).
- `.py` = 2 visible review passes + ruff clean + tests for the generalized admission (positive, negative: sha drift, live-tree path, missing reproducer). Serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, leave the fallback bundle and say so. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_vr5` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL
- vr4: `lsof` empty output under a sandbox with no process visibility proves nothing — the apply's gate must fail closed if `pgrep`/`ps` are unavailable (vr3's executor: verify it does; if it does not, add the refusal + test).
- vr3: the 26 rows it left were certificate-INCOMPLETE; 20 of them are now these rows because vr4 completed the chains — do not re-inventory, consume vr4's ledger.
- Moving bytes between the two SSDs creates no capacity (vr4 dead-end); this is deletion of rebuildable raws only.

## OPTIMAL FORM
- Reference form: vr3's executor and memo (`ff1095d49`) — full-certificate rows, journaled unlink, post-apply audit. SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no size-only admission; no skipped reproducer revalidation; no apply inside the sandbox.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
The generalized planner (code + tests), `ddm_vr5_reclaim_plan_20260909.jsonl`, the apply command line for MAIN, memo `.omx/research/ddm_vr5_generalized_certified_raw_reclaim_20260909.md`. Commit via the serializer. End with the live frontier line.
