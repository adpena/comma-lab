# ddm_sw1 portable paths, local-information scrub, and secret protection

**Disposition:** PARTIAL, evidence-complete, and **UNCOMMITTED**. The bounded implementation and
scrub are present in the shared worktree. The main-checkout serializer could not write Git objects
under the managed sandbox. A clean writable fallback clone reached the pre-commit hook, which then
refused three CI-blind MLX tests because this session has no Metal device. No hook or review override
was used. The exact patch is retained at
`$PACT_TIER2/ddm_sw1_20260820/patches/ddm_sw1_portable_paths_secrets_scrub.patch`
(101,888 B, SHA-256 `e8c9b955484f86ee277d747a39c85d4c50b10dec4112782947bbbeb3ff33c221`).

This arm did not run a scorer or exact evaluation, did not materialize a contest payload, and did
not move the frontier.

## Executive security verdict

The current changed surface contains no adjudicated credential or concrete private IP after the
scrub. Four final gitleaks findings are generic-rule false positives: two preflight token-name
constants and two research-prose phrases. The full history scan did find **16 real credential-shaped
findings** (12 GCP API-key detections and 4 JWT detections) in one historical commit,
`a3237dd6b7d45aee0baca4cdfc0697b863d0c466`, across three deleted browser-console logs:

- `.playwright-mcp/console-2026-04-10T15-52-51-364Z.log`
- `.playwright-mcp/console-2026-04-10T15-53-02-860Z.log`
- `.playwright-mcp/console-2026-04-10T16-26-35-872Z.log`

This is a P0 rotation/revocation obligation. The values were not reproduced in this memo. History
was not rewritten. Until the credential owner confirms revocation, these must be treated as exposed.

The configured fleet inventory is ignored by Git. None of its five exact configured IP values occur
in tracked or nonignored files after the scrub. Four of five short hostname aliases produce 1,229
matches; those aliases are common enough that the receipt is a candidate list, not proof that every
match is private fleet information. Hostname hygiene is therefore not closed.

## Phase receipts

| Phase | Disposition | Executed scope and result | Receipt |
|---|---|---|---|
| P0 census | PASS | Commit `d9af7d0f...`; 43,967 tracked files; 8,408 union matching files; 4,776 operator-home files / 70,795 occurrences; 5,868 mounted-volume files / 214,679 occurrences. Classes: a=921, b=4,953, c=2,534. | `$PACT_TIER2/ddm_sw1_20260820/receipts/public_repo_hygiene_census_summary.json`, 2,527,924 B, SHA `eac1e3e715f242982fb589e5f97bbf8132d3ab981e9ab9af7d19262ba7f95061`; rows JSONL, 5,303,717 B, SHA `2f694ccb1b5380a03bb0455a8eb763e47d95c208433adbaad0165e8d94b30360` |
| P1 resolver | IMPLEMENTED, UNLANDED | Extended `tac.payload_retention` rather than building a twin. Runtime roots resolve from `PACT_TIER1`, `PACT_TIER2`, and `HOME`, with current-machine defaults only at the waived resolver boundary. Durable retention rows carry portable paths. Relative paths remain relative. | Code plus 198-test focused receipt in this memo |
| P2 sweep | PARTIAL, DECLARED SCOPE | Sensitivity-ranked subset: 11 mapped files, including canonical retention writers, campaign/runner callers, the exact fleet-IP memo, and the backend runbook. Removed 64 legacy literals; added 73 placeholders. The one added local literal is the guard's own detector regex with a specific waiver. | `$PACT_TIER2/ddm_sw1_20260820/receipts/public_repo_path_mapping_receipt.json`, 4,613 B, SHA `f5ae4409a4cd37d8fecc7b08b9c581eb9514b9c63deec04e21a91df4d8dac3c3` |
| P3 staged guard | IMPLEMENTED, UNLANDED | Warn-only staged changed-blob check for operator-home and mounted-volume paths. Same-line waivers require a non-placeholder rationale. A planted violation fired exactly once and was removed; the final 15-file candidate surface is clean. | Retained control `$PACT_TIER2/ddm_sw1_20260820/retained_controls/ddm_sw1_absolute_path_positive_control.py`, 37 B, SHA `393326c72509033916fa05faa2f9e9b6fe88fce1d4775160221e0f90743ed9a6` |
| P4 current secrets | PASS WITH FALSE-POSITIVE ADJUDICATION | Nonvacuous live-rule control fired. Full tracked/nonignored snapshot: 44,005 files, about 1.72 GB plus one separately scanned 58 MB file. Initial scan: 98 findings (94 generic, 4 fleet-IP rule); one concrete resolver IP was scrubbed, three were generic CIDR labels. Final changed-surface scan: four generic false positives, zero credential/private-IP findings. | Control report SHA `10f6350ac44bd4277cbfab94c4117cb4a50e6193b170aefbd1ea4c141a96212f`; current report SHA `b63199e4fec20c97bc967d33cb54fa083b178ab5549d9903c3f79d58a0641e64`; final changed report SHA `fef17511ae0b54133c938cb5bc73b4041d7b2a664654521a3b99b999331799de` |
| P4 history | P0 ROTATION BLOCKER | Report-only scan of 15,339 commits, about 2.14 GB: 20,764 detector rows = 20,688 generic, 60 fleet-IP, 12 GCP-key, 4 JWT. The 16 real credential-shaped rows are isolated above. Git warned that exhaustive rename detection was skipped for an oversized diff; content diffs were still scanned. No rewrite. | `$PACT_TIER2/ddm_sw1_20260820/receipts/gitleaks_history.json`, 28,738,700 B, SHA `5c4acf9b334c12f578f62bf6f041e3464c8c2402761866f60e6cdea316ab2825` |
| P4 fleet | PARTIAL | `fleet.local.toml` exists and is ignored. Ten inventory coordinates scanned (five hosts, five IPs); zero IP matches, 1,229 short-alias matches. Values are absent from the receipt; only coordinate hashes and match locations persist. | `$PACT_TIER2/ddm_sw1_20260820/receipts/fleet_coordinate_scan.json`, 138,171 B, SHA `a7689ed67d51c08c840f6e3e369e1eaf48616330b85560ac17025406a1cad688` |

## Census interpretation and scope reduction

The prior-law prediction that at least 80% of hit files would be class-b memos was falsified. Literal
class b is 4,953 / 8,408 = 58.9%. The `.omx/research` directory is 6,984 / 8,408 = 83.1% of matching
files, but many of those are class c because their exact content SHA is consumed elsewhere. Classes
b+c together are 7,487 / 8,408 = 89.0%.

The complete 2,534-row class-c do-not-touch list, including every discovered pinning consumer, is
stored in the census summary under `class_c_rows`. The load-bearing code hit was
`src/tac/checkpoint_retention.py`, SHA
`b809cee5568445e930a69b3ebc1bebc5e6fcc209e16d75fb0e4b1640641c235c`, pinned by
`.omx/research/ADVISORY_sdf_postclosure_delta_watcher_ep25_curriculum_costate_20260710.md`.
It was restored byte-identically and remains unmodified. The protected submission, generation, and
upstream surfaces were not edited.

The sweep therefore did not pretend to close all 8,408 historical files. It closed a bounded,
consumer-checked subset and installed the structural staged guard. The migration taught every
changed code consumer to resolve placeholders in the same change; no memo parsed by a changed caller
was rewritten without that consumer path.

## Structural implementation

- `src/tac/payload_retention.py` owns the resolver, tier waterfall, portable formatter, and durable
  retention-record projection. Immediate consumers retain a runtime-resolved path; public JSON gets
  the placeholder form plus byte count and SHA.
- Canonical retention writers in `ddm_hm1`, `ddm_lr2`, and `ddm_lv1` now persist portable custody
  rows while continuing to keep every measured payload.
- The NeRV campaign planner/builder, compact renderer runner, and final-rate queue use the same tier
  surface. The campaign artifact stores placeholders; runtime entry points resolve them.
- The staged guard reads the actual Git index blob, not a cleaned worktree substitute, and skips
  binary blobs fail-closed with respect to text parsing rather than crashing.
- `tools/public_repo_hygiene_census.py` emits the classified census, mapping receipt, and hashed
  fleet-coordinate scan. SHA pins are found by content digest across every tracked text consumer,
  rather than by filename heuristics.

## Positive controls

The absolute-path control planted one disallowed changed-file line, observed one violation, then
removed it. The retained 37-byte fixture is identified in the phase table.

The secret control used a synthetic GitHub-PAT-shaped value matching a live gitleaks rule, not an
allowlisted documentation example. It returned the leak exit code and one finding, then was removed.
The retained fixture is
`$PACT_TIER2/ddm_sw1_20260820/retained_controls/ddm_sw1_gitleaks_positive_control.env`,
54 B, SHA `e73ccf25f7acd6ec4f9e97d0b619684d1b1ac121cdc26ac14e312f90b3e064f8`.

## Verification and landing boundary

- Focused resolver/guard/census/campaign/frontier tests: **198 passed in 60.30 s**.
- Final core hygiene tests: **25 passed in 0.44 s**.
- Broad caller suite: **598 passed, 6 xfailed, 4 failed in 797.01 s**. Three failures explicitly
  report no Metal device. The fourth is an unrelated SNeRV admission test blocked by two existing
  source-contract prerequisites; none of this arm's changed lines own those blockers.
- All 13 changed Python files compiled. Selected import/order/unused/modernization lint checks passed.
  Two real review passes were recorded for every changed Python file.
- The serializer in the shared checkout failed before commit because Git could not create an object
  database temporary file. The clean fallback clone passed the serializer's staging checks and
  reached the hook, but the hook refused three CI-blind MLX tests on the unavailable Metal device.
  No commit exists; the shared index remains untouched.

## RECALL EVIDENCE

Searched the full research/index/state/code surfaces using the queries `PACT_TIER1`,
`portable path`, `gitleaks`, `fleet.local.toml`, `absolute path`, `ddm_sw1`, `secret scrub`, and
`fleet info`; also queried the canonical-equations registry for path, storage, secret, fleet, and
retention terms. Sources included `CANONICAL_RESEARCH_INDEX*`, the full frontier DAG, the operator P0
ledger, the canonical frontier pointer/hot state, `checkpoint_retention.py`, and
`payload_retention.py`.

Beyond the charter seeds, recall found the exact SHA consumer for `checkpoint_retention.py`, the
already-settled pointer-only law against hardcoded frontier literals, and a newer owned frontier than
the common contract's stale line. That changed the work in three ways: the pinned writer was restored
instead of migrated, the census pin detector was generalized to all content-SHA consumers, and this
memo reports the live pointer rather than the charter literal. No storage-specific canonical equation
superseded the charter's tier waterfall.

## LIVE-HYPOTHESES

- Most of the 1,229 fleet-alias matches are ordinary short-word collisions, because exact configured
  IP matches are zero and only four short aliases account for the entire count. A semantic owner-aware
  adjudication should sharply reduce the real hostname set.
- A second sensitivity-ranked sweep can remove substantially more class-a breakage without touching
  pins, because only one load-bearing canonical writer in the selected surface proved SHA-pinned and
  the resolver now exists for its eventual superseding landing.
- The prepared landing should pass unchanged on a Metal-visible operator session, because 598 broad
  tests passed and both focused runs were green here; every hook failure named device unavailability,
  not a changed-code assertion.

## DEAD-ENDS

- A one-shot corpus rewrite is closed: 2,534 matching files are protected or content-SHA-pinned, so a
  global mechanical substitution would break custody.
- Editing `checkpoint_retention.py` in place is closed: its exact SHA has a named public consumer;
  supersede it or leave it byte-identical.
- Treating gitleaks' old allowlisted example as a positive control is closed: it is vacuous. The live
  PAT-shaped control is the replacement.
- Git history rewrite is closed: the real historical secret cure is rotation/revocation, not changing
  pinned public commits.
- Landing from this managed session is closed until both Git object writes and Metal-only hook tests
  are available; neither the shared-checkout failure nor the fallback-hook failure may be called a
  commit.

Own-vehicle frontier: **rc2_composed — S 0.14827847122030852 @ 180,456 B [contest-CUDA T4, n600]**;
this hygiene arm measured no score row and left the pointer unchanged.
