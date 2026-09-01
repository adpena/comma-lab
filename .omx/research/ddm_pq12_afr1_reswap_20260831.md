# PQ12 AFR1 packet re-swap — DONE, frozen and not published

Date: 2026-08-31 · Owner: `ddm_pq12` · Cost: **$0** · Scorer work:
**none** · `verdict_scope`: **INSTANCE** — this AFR1 archive, its enumerated
38-file receiver, and the generation-7 packet.

## Answer first

**SWAP_DONE_FROZEN.** The packet moved from rc2 to AFR1 and now binds archive
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`
at 180,002 B through runtime tree
`6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58`.
The stager verified 38/38 evaluated runtime rows, the census is clean, borrowed-
substrate accounting §11 is appended, and the strict compliance chain was
re-bought. No hosting, upload, push, PR, public comment, network post, scorer
launch, or `upstream/` mutation occurred.

Exact `[contest-CUDA]` T4 n600 arithmetic consumed from the retained authority
row:

`0.14797617125559104 = 0.020139 + 0.007981227975693965 + 0.11985594327989708`.

The evaluator's displayed `0.15` is not the claim.

## Stage receipts

| Stage | Measured result | Retained receipt |
|---|---|---|
| custody + seal | archive 180,002 B / `cbb8d928…`; tolerance-zero seal `a6450743…`; validation `SEAL_VALID` | `/Volumes/APDataStore/pact/ddm_pq12/receipts/CANDIDATE_SEAL_afr1_packet_target.json`, file SHA `ad430e30…` |
| runtime-only probe | 38/38 rows; re-derived tree `6cdfa27d…`; 39 source files including archive, 0 undeclared | `STAGE0_STAGING_RECEIPT.json`, SHA `524fdfcc…` |
| final stager | 38/38 rows; exact archive; nine declared packet documents; 0 source exclusions or undeclared files | `STAGING_RECEIPT_GEN7.json`, SHA `a5c0e405…` |
| final census | packet `CENSUS_CLEAN`; prep 31 flat / 0 nested; receipts clean | `CENSUS_GEN7.json`, SHA `5e7d02ee…` |
| strict compliance | 80 GREEN / 7 RED of 87 | `pre_submission_compliance.gen7.r2.json`, SHA `1628f3a9…` |

Frozen packet: `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/`.
Authority materialization and every receipt remain under
`/Volumes/APDataStore/pact/ddm_pq12/`; no payload was discarded.

## What changed from rc2

The full pointer ledger, rather than the charter shorthand, has five admitted
states after rc2: fx5 e1 180,386 B; dx2 180,368 B; gb1 180,215 B; lb1 180,083 B;
AFR1 180,002 B. `jt22` and `jt23` are closure/negative receipts in that ancestry,
not pointer moves. Total rc2→AFR1 change: −454 B and
−0.00030229996471747844 S.

All five mechanisms remain `ours-original` lossless coder/container decisions.
They introduce no learned artifact and do not change the borrow/own boundary.
AFR1's T4 row and rc2 emit the same 3,662,409,600-byte CUDA raw output, SHA
`6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883`,
so both distortion contributions cancel exactly and the whole delta is rate.

Two qualifications remain explicit. This is a re-decision over borrowed learned
content, not a new learned vehicle. The e2e rebuild VERIFIED label remains scoped
to packet generation 3; `compress.py` refuses AFR1 by exact SHA and names the
five missing post-rc2 stages rather than producing substitute bytes.

## Census falsifier

The first full census found three nested bytecode caches in the prep directory:
one created by this turn's compile check and two prior `verify_citations` caches.
Only those three explicit rebuildable `.pyc` files were removed, followed by the
empty cache directory. The re-run measured packet clean, prep 31 flat / 0 nested,
and receipts clean. AppleDouble sidecars created by APDataStore writes were also
removed only inside the new pq12 custody root immediately before each census.
No source, payload, receipt, or unrelated worktree file was removed.

## Compliance result and owners

The first run measured 75 GREEN / 12 RED. Five failures were stale compliance
inputs: the checker was asked to treat the private operator-input draft as the
policy answer, and the newest terminal lane row did not carry the checker's
accepted terminal spelling or both full hashes. The terminal run uses the factual
README as its private check input and a canonical appended harvested row; all five
are green. Seven real reds remain:

| Red check | Disposition | Owner | Why it remains red |
|---|---|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | `STRUCTURAL-RECORD` | MAIN policy adjudicator | raw authority emitter stamps its pre-adjudication blockers; raw receipt preserved |
| `contest_cpu_auth_eval_exists` | `RECORD-WITH-REASON` | MAIN scorer router | no AFR1 CPU score; same-lineage CPU was 0.0432 worse with about 21× pose degradation; none inherited |
| `submission_runtime_has_no_network_install_or_local_paths` | `HOLD-EVALUATED-RUNTIME` | MAIN runtime policy | evaluated `inflate.sh` contains the pinned Brotli bootstrap |
| `submission_runtime_imports_within_allowlist` | `DECLARED-NON-RUNTIME` | MAIN packet policy | staged `compress.py` imports repo `tac`; pretending it belongs to receiver closure would be false |
| `submission_runtime_tree_matches_auth_eval` | `ENUMERATED-TREE-PROVED` | MAIN packet policy | packet documents widen a fresh recursive walk; stager proves the exact 38 enumerated rows, but no separate equivalence proof was supplied |
| `public_scan_has_no_private_surface` | `HOLD-EVALUATED-RUNTIME` | MAIN runtime policy | authority-row `FX5_BUILD_MANIFEST.json` contains three provider-local paths; editing it would change evaluated bytes |
| `hosted_archive_manifest_supplied` | `BLOCKED-ON-OPERATOR` | operator | no hosting was authorized or performed |

No red is waived or relabelled green. The six non-hosting rows are recorded
properties whose cure requires changed bytes, a separate equivalence proof, a
new CPU policy requirement, or checker-policy adjudication. They are inputs to
the operator's publication decision, not hidden follow-ons owned by this arm.

## The one remaining gate

**BLOCKED-ON-OPERATOR-CONFIRM-AND-TEXT.** The operator owns the sole publication
gate: personally read the LLM-policy/full-stack inputs, decide which publication
path is acceptable including the seven recorded reds, write the entire PR
description and every public-facing comment in their own words with honest
authorship and borrowed-substrate disclosure, then give the one-line authority
for hosting and any PR/open/push action. Until that trigger fires, the packet
stays frozen and the hosted-manifest red remains real.

Consumer store after the trigger: operator-authored public text, hosted-archive
manifest, fetched-back byte-identity receipt, and final publication receipt.

## Git landing disposition

The one-line terminal AFR1 ledger cure landed normally through the serializer as
commit `60a9441468`. The packet commit then hit the managed-sandbox Git-object
write denial during `git add`; nothing was staged in the shared index. Its first
APDataStore fallback correctly refused with rc=19 because that tier had only
about 6.1 GB free and could not preserve the mandatory 40 GiB reserve. Retrying
the same serializer intent against VertigoDataTier produced the governed rc=17
bundle fallback under `/Volumes/VertigoDataTier/pact/ddm_pq12_commit_fallback/`.

Disposition: **BUNDLED-HANDOFF, not landed in canonical HEAD.** The bundle,
format-patch, and typed receipt preserve the exact 22-file intended packet
commit. No direct `git add`, direct `git commit`, index mutation, or fake landed
claim followed the denial. The final handoff names the retained leaf directory
and fallback commit after the bundle refresh that includes this paragraph.

## RECALL EVIDENCE

Searched the full corpus rather than only the charter seeds:

- `.omx/research/`, canonical research index, sub-0.15 DAG FEED blocks, hot
  state, and task ledgers with content queries `pq11|packet swap|rc2|AFR1`,
  `cbb8d928|tile48_groupbin8`, `RECORD-WITH-REASON|contest-CPU`,
  `runtime_tree_sha256|packet_census_guard`, and `e2e|NOT_EXPRESSIBLE`;
- `.venv/bin/python tools/list_canonical_equations.py --json` for current score,
  identity, and rate relations;
- source receipts for fx5, dx2, gb1, jt22/jt23, lb1, AFR1, UX1, the canonical
  pointer, and the actual authority/runtime manifest.

Beyond the charter seeds, four findings changed execution:

1. The exact rc2→AFR1 pointer ancestry is five moves: fx5 e1, dx2, gb1, lb1,
   AFR1. `jt22/jt23` are receipts, not moves. Accounting §11 uses the real ledger.
2. UX1's strict decode-resource scan counted 38 runtime files while the candidate
   seal counted 39 files because the seal includes `archive.zip`; stage 0 tested
   both denominators instead of treating them as a mismatch.
3. UX1's one-byte ZIP local-name idea remains a rider below the solo bar and was
   not silently folded into AFR1.
4. The CUDA result stored its returned artifacts as embedded strings. The
   canonical retention helper materialized them before the stager consumed the
   authority JSON; pointing the stager at the wrapper would have been false
   custody.

Scoped negative: did not find another live AFR1 packet generation or an exact
AFR1 CPU score in the searched research/state/custody surfaces.

## Measurement boundary

Measured this turn: archive/member bytes and hashes; 38 source and staged runtime
rows; tolerance-zero seal; staged enumerated tree identity; packet/prep/receipt
census; component arithmetic; and the real strict compliance result. Consumed,
not re-measured: the retained T4 score and raw-output identity. Not measured:
AFR1 CPU score, a fresh end-to-end rebuild, a clean CI run, public source
visibility, hosted-byte fetchback, or any new frontier candidate.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; pointer unchanged by this packet arm.`
