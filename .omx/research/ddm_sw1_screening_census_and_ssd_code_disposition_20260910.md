# ddm_sw1 screening census and SSD-authored-code disposition

Axis: `[macOS-CPU static apparatus / scorer-free]`. `score_claim=false`. No scorer job, payload mutation, archive evaluation, SSD deletion, or `upstream/` edit was performed.

## Result

The screening gate is complete and landed. The live census moved from 93 warnings among 375 charters (the pm2 seed snapshot was 93/371) to 0/375. All 93 warning rows are individually hash-bound `historical_finished_before_law` exemptions: 72 queue-backed charters have pre-law terminal receipts, including 71 final messages and one owning memo; 21 non-queue charters have pre-law committed owning memos. No row required a bulk exemption, rule weakening, or a declaration edit to a still-spawnable charter.

Commit `b4c23432f485ca0c0f1e9a1df81db48b95e50944` atomically landed the zero census, exact exemption ledger, fail-closed exemption validation, strict-by-default flip, Agent-path binding, generator, and tests. The exemption accepts only one exact row per repo-relative charter, exact charter/evidence SHA-256, the exact law commit/cutoff, a timezone-aware pre-cutoff completion time, and evidence below `.omx/research`; malformed, duplicated, stale, or post-cutoff rows fail closed. Explicit `TAC_SCREENING_LAW_STRICT=0` remains only the legacy process escape.

The original 182 cs1 `BLOCKED` code blobs classify as 171 real authored sources eligible for exact inert Git custody and 11 `RETAIN_WITH_OWNER`. The additional 61 general-debt extensions classify as 52 eligible and 9 retained. Thus the append-only cs1 disposition ledger contains 243 terminal attempt rows: 223 `BUNDLE_READY_MAIN_MUST_LAND` and 20 `RETAIN_WITH_OWNER`. No row was certified as generated output.

Fresh full audit before landing the blob copies: 286,984 code-like SSD files scanned; 1,596 distinct blobs absent from reachable Git; buckets A/B/D = 934/320/83; bucket C authored owed = 259. This supersedes the charter's 245 seed count: 17 fresh owed blobs were not in the 243-row cs1/other-extension plan, while one planned blob was no longer in the live owed set. Of the 223 bundle-ready rows, 222 are in the live 259 denominator. Therefore the exact post-apply count is projected, not measured, as 37; the current measured after-attempt count remains 259 because the active sandbox refused every blob insertion.

The nine serializer batches were each at most 25 files. Git returned rc=128, `unable to create temporary file: Operation not permitted`; the serializer returned rc=17 and retained nine bundles plus format patches. The combined 3,505,145-byte `landing.patch` has SHA-256 `c12aca2cc7ffde761d39e830e454fdea2df2cd5e1e22c4a71d1aec183548c702`. An isolated-clone verification applied all nine messages at base `951bcce1c788065120361413f54ac1a572284f9e`, produced nine commits, and recovered all 223 exact blob files. This proves the patch is applicable; it does not make the blobs reachable from the live repo ref.

The serializer's default fallback routed 7.9 MiB of receipts/bundles to `/Volumes/VertigoDataTier/pact/ddm_sw1/receipts/commit_serializer_fallbacks/` before the denial behavior was known. That was an SSD write outside the requested no-write boundary. Nothing was deleted or overwritten, the source SSD trees and their payloads were not mutated, and the new receipts are retained rather than silently removed. The later retry routed fallback locally. This caveat is recorded in the manifest and must travel with the handoff.

## Prediction audit

- Screening prediction passed: 93 historical finished rows is at least 70; true global screens = 0, at most 10.
- SSD classification prediction passed on the 182-row mandate: 171 are eligible real authored sources, at least 120; 0 generated-artifact certifications, at most 30; 11 retain with owner.
- The after-count target is not yet achieved on the live Git ref. It becomes 37 if the nine verified batches land and no further SSD drift occurs; a fresh audit is mandatory after apply.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus and arm receipts by content for `screening-law`, `TAC_SCREENING_LAW_STRICT`, `ssd authored`, `BLOCKED`, `retain-with-owner`, owner IDs, and terminal queue receipts; searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, task/state ledgers, `tools/`, and affected tests; ran `tools/list_canonical_equations.py --json`. Beyond the charter seeds, this found the exact Catalog #416 landing cutoff, terminal evidence for all 93 historical rows, the Agent-path deny-list parity surface, and live SSD drift to 259 owed with 17 new rows outside cs1's seed plan. Those findings changed the work from declaration editing to exact historical exemptions, added the Agent-path binding, and prevented the fresh 259 count from being mislabeled as the old 245 denominator. No score equation or scorer surface was consumed because this is apparatus-only work.

## Verification

- `42 passed` for the focused screening-law suite before integration.
- `71 passed, 1 deselected` for screening plus Agent-path parity; the deselected test is independently red from live correction-index horizon drift (`20260910` corpus beyond `20260909`), not from screening behavior.
- Final affected-suite repeat: `145 passed, 1 deselected` with that same unrelated horizon-drift test excluded by name.
- The larger run produced `144 passed, 2 failed`: one genuine Agent-path binding gap was fixed; the remaining clean-charter failure is the same unrelated live horizon drift.
- Ruff passed for `tools/codex_arm_queue.py`, `tools/agent_model_routing_guard_hook.py`, `src/tac/tests/test_ddm_pm2_screening_law.py`, `experiments/ddm_sw1_build_screening_exemptions.py`, and `experiments/ddm_sw1_disposition_ssd_code.py` after two visible review-tracker passes.
- Historical SSD Python review: 111 `.py` blobs, 420 Ruff findings across 59 files. Correctness-risk codes (`F821`, `B023`, or invalid syntax) caused four rows to remain owner-retained; style/hygiene-only findings were recorded, not erased. All 71 shell blobs passed `bash -n`.
- `git am` of the combined landing mbox in an isolated clone: PASS, 9/9 batches, 223/223 blob files.

## Artifact map

- Screening exemption ledger: `.omx/research/ddm_pm2_20260909/screening_law_exemptions_20260910.jsonl`
- Zero census: `.omx/research/ddm_sw1_20260910/screening_census_after.json`
- Fresh SSD audit: `.omx/research/ddm_sw1_20260910/ssd_audit_before.json`
- Append-only disposition rows: `.omx/research/ddm_cs1_ssd_code_certify_20260909.jsonl`
- Disposition summary: `.omx/research/ddm_sw1_20260910/ssd_disposition_summary.json`
- Apply manifest and verified mbox: `.omx/research/ddm_sw1_20260910/landing_manifest.json`, `.omx/research/ddm_sw1_20260910/landing.patch`
- Patch verification: `.omx/research/ddm_sw1_20260910/landing_patch_verification.json`

## Follow-on disposition

- `FIRED` — screening strict flip; owner `ddm_sw1`; consumer `tools/codex_arm_queue.py` and Agent spawn hook; trigger satisfied by zero census and commit `b4c23432f`.
- `QUEUED-WITH-FIRE-ORDER` — land nine exact-source batches; owner `MAIN with Git-object write authority`; consumer `.omx/research/ddm_cs1_ssd_code_certify_20260909.jsonl` plus the SessionStart SSD monitor; fire when the manifest hashes verify and the live repo accepts Git object writes.
- `QUEUED-WITH-FIRE-ORDER` — rerun the full audit and append commit-verified terminal rows; owner `MAIN`; consumer `.omx/state/ssd_authored_signal_audit_cache.json`; fire immediately after all nine batches are reachable.
- `FOLDED` — 20 retained rows stay with their named source owners in the cs1 ledger; consumer is the same ledger; reopen only when the row's named owner supplies a clean exact source or deterministic generated-artifact reproducer.
- `FOLDED` — 17 fresh concurrent rows remain under the consolidation monitor and their source owners; consumer is the fresh audit manifest; they were outside the fixed cs1 seed plan and must not be silently absorbed into this landing.

OWN-VEHICLE FRONTIER: S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]
