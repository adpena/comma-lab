# DDM FC1X — serializer fat-clone cure (#1302)

## Outcome

**COMPLETE AS TWO VERIFIED THIN-BUNDLE LANDINGS; MAIN consumption remains queued.**
The reachable denial fallback in `tools/subagent_commit_serializer.py` no longer
clones the Pact checkout. It now constructs the intended commit with isolated Git
plumbing against the checkout's object database as a read-only alternate, emits the
same rc=17 bundle/patch/receipt family, and refuses before any SSD write when either
the 64 MiB artifact cap or the canonical 40 GiB storage reserve would be violated.
A strict #1302 preflight guard prevents clone-based fallback from returning in the
serializer or its direct landing-tool sisters.

This was apparatus-only work: no scorer, Modal, archive mutation, or exact evaluation
was run, and the canonical frontier pointer did not move.

## Located path and incident receipts

The fat path was the existing `_author_bundle_fallback` branch in
`tools/subagent_commit_serializer.py`: on a main-checkout Git-object write denial it
ran `git clone --quiet --shared --no-checkout <repo_root> <scratch_repo>`. A bounded
census of the direct landing tools found no second fallback clone site. The legitimate
remote-public-intake clone in `tools/fetch_all_public_pr_archives.py` is outside this
landing/recovery scope and is preserved.

The incident is cited from its custody receipts rather than reconstructed:

- `.omx/research/arm_final_messages/ddm_bs3_born_small_resolved_carrier_20260826T203546Z.md:23,44`
  records the 8.4 GiB fallback allocation, reserve breach, and absence of a commit or
  bundle, and explicitly says not to rerun the full-checkout fallback.
- FB2's certified cleanup receipt removed the recoverable scratch while pinning tree
  `26b27dce163fa2be966b980aa651d8b828e83f1e`; its route-table context is
  `.omx/research/ddm_fb2_route_table_gb1_20260826.md`.
- The recurrent denial-class and proven rc=17 contract remain pinned by
  `.omx/research/ddm_hd1_apparatus_two_landings_20260826.md`, implementation commit
  `a08ea28d77`, and evidence commit `f3d6aba3e1`.

## RECALL EVIDENCE

- Governing surfaces read: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the common Codex
  contract, the FC1X charter, operating manual, live hot state, task-status ledger,
  canonical-equation registry, HD1/FB2 memos, and the BS3 final receipt.
- Exact search surfaces included `git clone`, `--shared`, `--no-checkout`, serializer
  fallback receipts, rc=17, and direct landing-tool invocation sites. No canonical
  equation governs this control-plane mechanism.
- The search beyond the supplied seeds changed the plan: HD1's supposedly separate
  “good path” was itself backed by the reachable checkout-clone implementation. The
  cure therefore replaced the construction mechanism while preserving HD1's external
  rc=17 and receipt contract; it did not delete a distinct dormant legacy branch.
- The prior-law `<100 changed lines` prediction was falsified. Preserving exact
  patch-intent and whole-file modes, receipts, rc=17, deterministic commit metadata,
  reserve-refusal receipts, and focused controls required 472 insertions/99 deletions
  in landing 1. The no-clone mechanism claim is real despite that size miss.

## Landing 1 — source repair

Commit: `dff62c8085dcfee4dfc25d59202b61928ba2afbf`

Base: `fa6eb2dab0cdc97524a0571e69d5af6d54890d72`

Owned files and post-edit SHA-256:

- `tools/subagent_commit_serializer.py` —
  `2172fe658f46256328147f0d856833151baadc23f31c4838cc958d0c4e39b0e3`
- `src/tac/tests/test_subagent_commit_serializer_bundle_fallback.py` —
  `a07ce5fc01fd9ff23bc8cd4a61b9894db3570b121f81a26d3ec1a252f9943ada`

The fallback uses `read-tree`, `hash-object`, `update-index`, `write-tree`,
`commit-tree`, `update-ref`, and `git bundle` inside an ephemeral bare Git directory.
The Pact worktree is never cloned or checked out. Success remains rc=17 with schema
`subagent_commit_bundle_fallback.v1`, the verified bundle, format-patch,
intended-tree patch, and per-file content hashes. The success receipt adds
`construction_mode=isolated_git_plumbing_no_checkout` and the cap/reserve facts.

The waterfall imports `DEFAULT_RESERVE_FREE_GB` and `bytes_from_gib` from
`src/comma_lab/storage_tiers.py`; it calculates the exact final bundle/patch bytes plus
a bounded receipt allowance before creating a requested SSD directory. If projected
free space falls below the canonical reserve, it emits a loud local small refusal
receipt and returns typed rc=19 without an SSD write. Oversize fallback artifacts use
the same rc=19 and a 64 MiB cap. Construction failures remain rc=18.

Retained landing bundle:

- `.omx/tmp/codex_runs/ddm_fc1x/landing1_dff62c8085.bundle`
- 6,572 bytes; SHA-256
  `1ba5034c17954d59873769ce0e230b9dcf5ce181f89a291c34ddddac5eb130af`
- `git bundle verify`: OK; contains `dff62c8085...`, requires `fa6eb2dab0...`.

## Landing 2 — strict class guard

Commit: `b2763c5eddfe3797920a1cf43b02f1eb45e1a7ff`

Parent: `dff62c8085dcfee4dfc25d59202b61928ba2afbf`

Owned files and post-edit SHA-256:

- `src/tac/preflight.py` —
  `f6f910964559a7f4bcf3396c65a4d5a98cd31b86a52d66f48e894b4f7cee08ad`
- `src/tac/tests/test_check_1302_no_clone_based_serializer_fallbacks.py` —
  `c725a9348c7d0658f4a97927b65e9deddceedfa9578675e5eda4efe42c9afbc2`

`check_no_clone_based_serializer_fallbacks(strict=True)` is wired into
`preflight_all`. It parses Python landing tools with the AST, scans shell sisters,
deduplicates nested command expressions, and rejects `git clone` in the serializer or
direct landing tools. A legitimate scoped use requires the same-expression waiver
`# SERIALIZER_FALLBACK_CLONE_OK:<substantive rationale>`; placeholders fail closed.

Retained incremental landing bundle:

- `.omx/tmp/codex_runs/ddm_fc1x/landing2_b2763c5edd.bundle`
- 3,932 bytes; SHA-256
  `2bdc167bbd3e9cedf4e2a3278de73f30028d538318e7508ca0023070beaf11f8`
- `git bundle verify`: OK in the landing repository; contains `b2763c5edd...`,
  requires `dff62c8085...`.

## Executed controls

- Complete serializer family: **118 passed** in 37.08 s.
- Guard/serializer/integration focus: **45 passed** in 2.89 s.
- Positive controls executed: multiline synthetic serializer clone, direct Python
  sister clone, and shell sister clone all fired; placeholder waivers failed.
- Negative controls executed: current landing tools clean, legitimate public intake
  outside scope clean, substantive waiver clean, and normal serializer success path
  remained rc=0 without invoking fallback.
- Read-only-object denial control produced rc=17 with a verified bundle and matching
  receipt. Reserve control returned loud rc=19 before the requested SSD directory
  existed. Exact-patch-intent mode used the same clone-free construction.
- Ruff and `git diff --check` passed. Both Python files in each landing received two
  genuine review passes; targeted review-policy checks report zero violations.
- MG1 live regression receipt remained byte-valid and schema-compatible:
  `/Volumes/VertigoDataTier/pact/ddm_mg1_serializer_fallback/20260826T214246.831553Z-18638/receipts.jsonl`
  names schema `subagent_commit_bundle_fallback.v1`, rc=17 flow status
  `BUNDLE_READY_MAIN_MUST_LAND`, commit `d6d3309e1c5e3b4251f00a86c912b87c918e3a13`,
  and bundle SHA-256
  `eb6453dc875437d9b8a86cdf1dec5be74e60504936f7f6bae1a188fa7d9be88e`;
  the retained bundle verifies.

## Ledger closure

Task #1302 is completed by the two commits above with green test status. Main is
responsible only for consuming the ordered bundles; that custody step does not reopen
the repair or prevention verdict.

GESTALT-DELTA: denial recovery is no longer a hidden second checkout competing with
payload custody for SSD reserve; it is a bounded object-plumbing transaction whose
small evidence is either retained as rc=17 or loudly refused as rc=19 before SSD
mutation, and a strict executable guard now preserves that distinction.

## LIVE-HYPOTHESES

- Dynamic clone construction outside the explicit direct-sister list could evade this
  bounded guard. It remains plausible because Python can assemble argv from variables,
  but no such path was found in the recalled landing-tool surfaces.
- The recurrent Git-object denial is imposed by the managed sandbox rather than Unix
  ownership or mode. That remains plausible because the object directory was readable
  and mode `0755` while writes failed with `Operation not permitted`; FC1X did not test
  sandbox policy itself.

## DEAD-ENDS

- Full-checkout clone fallback: closed by the BS3 incident receipt, source repair, and
  strict positive control. Do not retry it on either SSD.
- Prevention-only closure: closed as insufficient by the two-landing rule; landing 1
  repairs and landing 2 prevents.
- A locally invented reserve or post-write capacity check: closed; the implementation
  imports the canonical waterfall constant and refuses before the first SSD mutation.
- Treating remote public-repository intake as serializer recovery: closed by scope;
  that legitimate clone is not aimed at the Pact checkout and remains untouched.

OWN-VEHICLE FRONTIER: GB1 — S 0.14811799921260607 @ 180,215 B
`[contest-CUDA T4, n600]` — UNMOVED.
