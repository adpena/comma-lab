# DDM SF2 structural fix wave — 2026-09-01

## Result

All three recurrence classes are **CURED-STRUCTURAL** at their named populations. This
was a scorer-free apparatus landing; it launched no training, scorer, Modal, or exact
evaluation work and did not modify `upstream/`.

- **Class A — CURED-STRUCTURAL.** Both certify-and-move implementations now measure
  destination metadata representability, refuse a lossy direct tree, tar-wrap when
  required, verify a restored form against per-path content SHA-256 + exact modes +
  symlink targets, and re-read the source before retirement. The serializer now refuses
  immediately when the live venv's `python3` does not resolve/run or `ruff` does not
  execute.
- **Class B — CURED-STRUCTURAL.** The existing shared `tools/premise_lint.py` surface now
  owns one prefix-match/divergent-tail matcher. The keeper charter lint and staged
  research-memo serializer both consume it and refuse rather than warn.
- **Class C — CURED-STRUCTURAL.** The keeper's actual Codex charter scaffold carries the
  binding detached-process clause. `tac.subagent_contract.standard_contract` was not
  edited because source tracing found that it does not compose keeper-generated Codex
  charters.

No score moved. These are means, not frontier progress.

## Class A — measured mechanism and cure

The charter's instance-level root-cause hypothesis was wrong. The affected `uv-envs`
move did not target APDataStore: `vr2_local_coldstore_move_ledger_20260831.jsonl` rows
101–104 show `/Users/adpena/pact_cold_store/pact/pact/uv-envs` on local APFS. Its census
counted six symlinks, but the pre-cure mover only copied regular-file relpaths and
recreated them with `open(..., "wb")`; it neither retained link targets nor restored
modes. Thus the actual loss was in mover logic, not destination ExFAT.

The AP hypothesis was still tested directly. The mounted APDataStore is ExFAT/FsKit.
On that live filesystem, try-create/readlink succeeded, while requested file modes 0640
and 0751 and directory mode 0750 all read back as 0700. Therefore the broad claim
"ExFAT cannot represent symlinks" is refuted for this mounted implementation; exact
POSIX modes are measured unrepresentable there.

The cure has four structural legs:

1. `tools/vertigo_certify_move.py` inventories exact file/directory modes and symlink
   targets, probes the selected destination, and selects `direct_tree` only when every
   required value is exactly representable.
2. An unrepresentable payload is retained as a PAX tar plus content and metadata inner
   manifests. The cert records the tar SHA and an explicit restore command.
3. Both tar and direct paths independently re-read content and metadata after
   materialization; source retirement is unreachable unless source, destination or
   restored form agree.
4. `src/comma_lab/artifact_retention.py` consumes the same fidelity implementation, so
   its two cleanup-tool consumers and the AD2 direct consumer cannot bypass the cure.
   The AD2 cold-store reader recognizes and revalidates the tar-wrapper representation.

The actual filesystem control is retained at
`/Volumes/APDataStore/pact/ddm_sf2/controls/RESULT.json` (10,240-byte tar at
`/Volumes/APDataStore/pact/ddm_sf2/controls/metadata_source.tar`). The result SHA-256 is
`9eb5ce8413f5f07ab24ccd52eb0cc6932d8eaa0ea80aa2f75bbd672503738725`; the retained tar
SHA-256 is `8617eb4811c29ef5187fc7433577da224494ec1046efd6d34cf589b026748fcb`.

Executed controls:

- **Positive/direct refusal:** a retained tree with one relative symlink, a 0751
  executable, a 0640 file, and 0750 directories produced three exact mode blockers on
  live APDataStore. Direct move was refused.
- **Positive/tar round trip:** the same tree tar-wrapped to APDataStore and restored on
  Vertigo with 2/2 content rows, 5/5 metadata rows, exact 0751/0640/0750 modes, and the
  original symlink target.
- **Negative/direct:** a clean tree on metadata-capable Vertigo produced zero blockers,
  copied directly, and matched both content and metadata manifests.
- **Sister positive:** the shared retention mover's cross-device mismatch control chose
  `tar_wrap_verify_delete`, retained the inner manifest/restore receipt, and restored
  the symlink and 0751 mode.
- **LOUD canary:** the live venv passed; a synthetic missing `python3` target plus
  non-executable `ruff` produced both blockers.

## Class B — shared refusing lint

The matcher examines every standalone hex run of at least 16 characters. A token fires
only when it shares at least eight leading characters with a canonical full SHA or a
full SHA pinned in the same document and then diverges. Exact values, honest prefixes,
and unrelated hashes do not fire. Keeper charter lint fails closed if the shared helper
is unavailable; the serializer returns a dedicated refusal before staging or commit.

Executed controls:

- **Positive:** the actual sfp1 `cbb8d9283f4352…` line produced one refusal that showed
  both the observed and canonical values.
- **Negative/canonical:**
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` produced zero
  findings.
- **Negative/unrelated:** a random unrelated 64-hex token produced zero findings.
- **Same-document positive:** two synthetic full pins with an eight-hex shared prefix
  and divergent tails produced one finding even with no pointer pins supplied.

## Class C — session-independent heavy compute

Source tracing identified `_CHARTER_TEMPLATE` in `tools/codex_arm_queue.py` as the live
input to the keeper's `scaffold` subcommand. The generated clause says that any single
compute step projected above 30 minutes must use `nohup` + `disown`, a pidfile,
crash-resumable stage checkpoints, and a durable done receipt; the arm monitors it and a
successor or MAIN harvests it. An in-session multi-hour compute loop is forbidden.

The real scaffold command emitted
`/Volumes/APDataStore/pact/ddm_sf2/controls/scaffold_control.md`, SHA-256
`359a2670e8d75bcbacaf63a6a33e99b39b085627cb561c72d5b1d273686c075c`, and the executed
assert found the whole clause. This is honestly **scaffold-only coverage**: no additional
hand-authored-charter prose lint was added because the existing optimal-form lint does
not cheaply parse shell execution topology.

## Denominators and boundaries

- **A / CLASS:** 2/2 certify-and-move implementations covered. Their 4/4 live consumer
  surfaces are the Vertigo CLI, compact-experiment cleanup, DQS1 local-first cleanup,
  and AD2 retained-measurement cleanup. Other ordinary copy/staging routines are not
  source-retiring certifiers and are outside this population.
- **B / CLASS at lint ingress:** 2/2 live ingress surfaces covered: keeper charters and
  new/edited `.omx/research/*.md` files serialized through the mandatory commit path.
  The bounded corpus currently has 9,491 research Markdown files, including 272 charter
  files; this landing does not claim a retrospective clean sweep of all 9,491. Each
  crosses the guard when newly authored or next edited.
- **C / CLASS at keeper scaffold:** 1/1 keeper-generated Codex charter composition
  surface covered. Existing 272 charters are not retroactively rewritten, and
  hand-authored charters remain outside the scaffold-only guarantee.
- The controls establish filesystem fidelity and apparatus behavior only. They do not
  establish score, candidate correctness, contest runtime, or frontier movement.

## RECALL EVIDENCE

The recall searched the full bounded corpus before editing:

- Research and receipts: content queries for `#1165`, `certify-and-MOVE`,
  `vertigo_certify_move`, `uv-envs`, `pact_cold_store`, `symlink`, `exec bit`,
  `transcription`, `sfp1`, `rxc1`, `scaffold`, `nohup`, and `done-receipt` across
  `.omx/research/` and arm final messages.
- Exact source/callsites: `rg` over `tools/`, `src/`, and `experiments/` for
  `execute_retention_plan`, `_copy_verify_then_delete`, `copytree`, the keeper template,
  and `standard_contract`.
- Canonical state: `.omx/state/main_hot_state.md`, the canonical frontier pointer,
  canonical task-status rows touching #1165/#1169, and the live queue/consumer stores.
- Math and graph surfaces: `tools/list_canonical_equations.py --json`,
  `CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks, design/SPEC text, and the
  task ledger using storage/metadata/SHA/detach terms.

Beyond the charter seeds, recall found the shared artifact-retention sister mover with
three consumers. That changed the plan from a one-tool patch to a 2/2 implementation,
4/4 consumer cure. Recall also found that the actual uv-env destination was local APFS,
which refuted the charter's instance mechanism and redirected the cure from a filesystem
name check to exact destination probes plus fidelity verification. No relevant canonical
equation or score-bearing DAG law was found in the searched scopes; no equation/DAG
mutation was warranted for this apparatus-only landing.

Pinned source evidence: pre-cure `tools/vertigo_certify_move.py` SHA-256
`6ba92499ad3ba9ce9206fba35da521b5f6828bc117a72801e6447e1db4cc520d` per
`ddm_vr2_vertigo_reclaim_round2_20260831.md`; the local move ledger SHA-256 is
`4b31927bb8b56a03cf6e44b1ab9353bed0ebc1ad062a1e144457b3449c36302d`.

## Verification and review

- `py_compile`: 6/6 edited implementation modules passed.
- Ruff: all edited implementation files and focused tests passed; the keeper's broader
  file was also checked for fatal parse/name errors. One unrelated pre-existing style
  finding at `test_codex_arm_queue.py:810` was not changed.
- Pytest: **131/131 passed** across Vertigo mover, shared artifact retention, keeper
  queue/scaffold, and serializer controls.
- Two genuine review passes were recorded for all 10 edited Python files after the final
  content stabilized. Pass 1 reviewed source-retirement ordering, provenance, failure
  recovery, and live consumer routing. Pass 2 adversarially exercised real/synthetic
  representability, tar restore, divergent-tail positives and negatives, venv failure,
  and scaffold output.

## MAIN disposition

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN storage apparatus; consumer store: the next
  `vertigo_certify_move` or artifact-retention execution receipt; fire trigger: the next
  source-retiring cold move. Confirm its plan records `direct_tree` or `tar_wrap`, exact
  destination capability results, and a true metadata/content fidelity receipt before
  accepting reclaimed bytes.

Own-vehicle frontier: **S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]**, afr1
archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` — **UNMOVED**.
