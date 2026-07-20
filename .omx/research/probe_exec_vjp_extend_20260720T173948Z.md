# Probe execution + VJP custody extension (2026-07-20)

`research_only=true` · `$0 LOCAL` · `score_claim=false` · pointer `0.1910828242 [contest-CPU] UNMOVED`

## Verdict

Leg A is measured on the full real n600 target cache, but only P2 composes through the registered surfaces. P2 is `FALSIFIED_FORMULATION`: tropical principal costs 137 bytes versus Aurenhammer LP 134 bytes under the exact same PDW2+Brotli-q11 coder (zero-sum is 131). P1 and P3 are `BLOCKED_NOT_MEASURED`, not negative family verdicts: surface 1 accepts an already-defined RGB proposal but surfaces 1-3 do not define the missing class-probability/rank-4-feature-to-RGB pullback required to create bytes, parse back, and invoke a fresh frozen CPU-Torch hard oracle.

The P1 target-only decomposition remains useful but non-authorizing: the n600 sparsemax exact-one-hot fraction is exactly 0.9733309173583984, identical to the preregistered value, with full per-class and margin-stratum rows in the receipt. No hard-accept, exact-call, or candidate-byte count was invented for P1/P3.

Leg B passed its launch gate. Fresh pairs 26-28 took 33.38 seconds total (11.1267 s/pair), peak RSS was 2,670,051,328 bytes, and the 576-pair projection is 6,408.96 seconds / 1.7803 hours. The memory governor admitted a 3.0-GiB projected workload with 60.493 GiB modeled headroom. The job is detached, SSD-resident, atomic per pair, and restart-proven: launch 1 was intentionally terminated after pair 29; launch 2 observed a partial prefix, left pair 29's 151,686,708 bytes, mtime, and SHA-256 unchanged, skipped it, and completed pairs 29-40.

One exact blocker remains visible: pair 11 refuses the frozen `cached_winner_native_rival` arrangement because its fresh native winner differs at one pixel. The campaign isolates and preserves that scoped refusal, then continues independent chunks. Therefore this handoff is `LIVE_RESUMABLE_EXTENSION`, not n600 completion, and the 38,077-candidate Fisher EV ordering remains correctly un-emitted.

## Durable artifacts

- Probe manifest: `.omx/research/probe_exec_vjp_extend_20260720T173948Z/regmax/manifest.json`, SHA-256 `4993ed2396f6ee4de862454f739d0e385acfc7b454ea4a41561e4d928ae6abbb`.
- Timing receipt: `.omx/research/probe_exec_vjp_extend_20260720T173948Z/vjp_timing_receipt.json`.
- Restart proof: `.omx/research/probe_exec_vjp_extend_20260720T173948Z/vjp_resume_proof.json`.
- Watcher: `.omx/research/probe_exec_vjp_extend_20260720T173948Z/vjp_extension_watcher.md`.
- Live SSD campaign receipt: `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json` (mutable until terminal).

## Verification

- `ruff check` and `py_compile`: clean for both new tools and tests.
- Probe executor tests: 7 passed; extension-driver tests: 4 passed.
- Full n600 probe execution: 46.567 seconds, git head `be7bac503c4d0c51661858337520953c28bb537d`.
- Two review-tracker passes were recorded for each Python landing before serializer commits.
- No paid/GPU/evaluator launch; no frontier pointer edit.

## STORES CONSULTED

- delegated authority prompt and `AGENTS.md`/`CLAUDE.md` operating contracts
- `.omx/research/prereq_surfaces_flush_20260720T171050Z.md`
- `.omx/research/prereq_surfaces_flush_20260720/manifest.json`
- `.omx/research/erm_2607_10128_crosswalk_20260720T154953Z.md`
- existing VJP manifests and immutable sidecars under `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719`
- canonical per-arm and broadcast inboxes through broadcast UTC `2026-07-19T19:48:01Z`

## MAIN landing review required

MAIN must review `b1aac4eb47..codexwt/probe_exec_vjp_extend_20260720T172411Z`, especially (1) the typed-pullback blocker boundary for P1/P3, (2) the P2 same-coder byte comparison, (3) source-manifest uniqueness and pair-11 refusal semantics in the campaign driver, and (4) the live campaign's terminal receipt before any n600/Fisher-EV claim. Merge review is mandatory; this branch is not source of truth.
