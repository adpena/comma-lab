# ddm_scg2 — historical sj1 custody successors and named runtime digests

# FORMALIZATION_PENDING:seal custody apparatus — no measured law

Two successor seals are persisted beside the unchanged historical sj1 seals. Both validate
`SEAL_VALID` against the live rc2 pointer. This is custody apparatus (`score_claim=false`),
not a score improvement. Both fire dry-runs passed without a waiver, with complete source
snapshots and successful import verification. scg1 ITEM_7 is completed in the canonical
task ledger; main-branch landing of the retained bundles remains separately queued.

## Retained custody

Root: `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/`.
Evidence directory: `scg2_custody_20260909/` beneath that root.

| Candidate | Successor seal | Already earned T4 score | Historical Modal call |
|---|---|---|---|
| joint | `SEAL_ddm_sj1_token_predistortion_joint_contest_cuda_successor_20260909.json` | 0.1398140172839628 at 180,904 B | `fc-01M1T6TCW2JS1JEW5CSZH3FVBY` |
| pass3 | `SEAL_ddm_sj1_token_predistortion_pass3_contest_cuda_successor_20260909.json` | 0.13900437796841966 at 181,645 B | `fc-01M1TFD35EPY2YZHNV3VKJP6MG` |

These are existing `[contest-CUDA T4 n600]` rows, re-derived from the retained
`MODAL_REMOTE_RESULT.json` components as `100*d_seg + sqrt(10*d_pose) + 25*B/37545489`.
They are not new measurements by this arm. Each successor records `supersedes` (the old
seal's canonical document hash), `supersedes_file_sha256` (the actual historical file
hash), `supersedes_path`, and `already_scored` with source receipt SHA, call, components,
score and runtime definition. The new admit bar explicitly describes custody replay,
with the current frontier as smoke control; it does not invent an improvement forecast.
The original admission assertions remain in the historical seal.

`joint_current_smoke.json` and `pass3_smoke.json` contain the final 8/8 successful legs:
two public-path token-decode-reach probes and two public shell CUDA-gate probes per
candidate, including the rc2 frontier control. Bounds: 125 s declared, 100 s subprocess
probe. The direct outcomes are the existing producer's bounded, no-exception
`REACHED_TOKEN_DECODE` receipts, not completed decodes or scored outputs. Shell outcomes
are the receiver's nonzero `RuntimeError` CUDA gate.

The first joint run also passed 4/4 legs, but MAIN promoted rc2 during that run. The
old-control receipts remain in `joint_smoke.json`; the pointer-at-issuance assertion
prevented a stale successor. Joint was rerun against rc2. Pass3 already started against
rc2. Both final run-time pointer captures are retained.

## Digest definitions, verified at source

| Tree | `tac.candidate_seal.measure_runtime_digest` | `tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest` |
|---|---|---|
| joint | `4b871196ebc653ce987b0b29e826003abd9f0f202eef75f453f57aed2a47c959` | `9d51671dedd0b2b10b39903fba6883ba4438b587ab9e54097aa5ff1f7c8e404b` |
| pass3 | `2435dab86725ed8e3ca1bf55d72834272f44b4aa24ee343a4381ed4e1df9132a` | `6508d184e60ab3e3c63b05ed5f0f3bc3ff551408f38af1620f90c62c0b21783e` |

MEASURED `[macOS-CPU custody apparatus]`: both definitions reproduce their historical
values on both trees. They differ on 2/2 tested trees; this is not evidence of drift.

- VERIFIED-AT-SOURCE: `src/tac/candidate_seal.py::measure_runtime_digest` hashes sorted
  `(relative_path, bytes, sha256)` triples using its existing shipped-file exclusions.
- VERIFIED-AT-SOURCE: `experiments/modal_auth_eval.py::_expected_uploaded_runtime_tree_sha256`
  composes `experiments.contest_auth_eval._runtime_dependency_manifest` with
  `tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest`. The latter
  hashes the projected remote manifest, including the upstream evaluator and import metadata.
- Neither computation changed. `RuntimeDigest.to_dict` now names its definition; seal
  construction labels a private copy of validated legacy producer receipts. Explicitly
  wrong labels refuse; old unnamed seals remain readable. The protected sj1 producer
  was imported and reused, never edited or replaced.
- `tools/fire_modal_auth_eval.py::measure_fire_runtime_digests` records both values and
  definitions under `stage3_runtime_digests`; when a seal is supplied it also records the
  sealed value. Successor seals carry the same pair as `runtime_digest_cross_reference`.

## RECALL EVIDENCE

Read the whole charter, common contract and scg1 memo; checkpoint read found no predecessor.
Searched `/Users/adpena/.codex/memories/MEMORY.md` for `custody|serializer` and verified the
post-edit, repeated-hash serializer procedure against current source. Searched research
memos/receipts by content for `digest_definition|2435dab86725|6508d184e60a|public_entrypoint_smoke|SEAL_PUBLIC_SMOKE`,
and task status for `scg1.*ITEM_7|seal.*digest|runtime.*digest`. Searched the canonical
research index, `sub015_DAG_*` FEED material and `docs/` for
`seal|runtime.digest|public.entrypoint`; the broad historical DAG seal hits did not add
a competing custody procedure. The canonical equation registry JSON returned 477 rows;
the full JSON is retained in `equations.json`. No measured law is claimed for this apparatus.

Beyond the seeds, direct worker-source recall located the projection function in
`experiments/modal_auth_eval.py`, not the fire CLI. This changed the implementation to
compose the same two existing manifest helpers, with no new digest algorithm. Live
pointer/source recall found MAIN's rc2 move and the updated sj1 pointer lineage; this
caused a fresh joint control run instead of blessing a stale bar. The storage waterfall
selected APDataStore for bulky scratch and snapshots; small successor receipts remain
beside the originals on Vertigo.

## Verification and boundaries

122/122 tests passed across candidate-seal, pin-consistency, fire-axis, claim-exemption,
done-receipt and lane-name suites. `ruff check` passed. Two genuine post-edit review passes
covered 43 seal entities, 15 fire-tool entities and 48 test entities on each pass; tracker
marks rescan the AST. The reviews checked immutable inputs, unchanged hash computation,
legacy compatibility, wrong-definition refusal, and the build/validate/fire seam.
`verification.json` and `execution_provenance.json` retain the scope and content hashes.
`joint_fire/FIRE_MANIFEST.json` and `pass3_fire/FIRE_MANIFEST.json` both record
`dry_run=true`, `PRESENT_AND_SEAL_VALIDATED`, both digest definitions, and no source-snapshot
or import-probe failures. The same verified snapshot served both dry-runs: 8,331 mounted
files, 642,821,717 logical bytes, digest
`434247ba363d9c6e369aa4efb1992fb85fcb7fae5160897879680148bbd28c28`.
Catalog #344 returned no findings in strict mode; receipt `catalog344.json`.

Smoke invocations ran sequentially in the foreground. Subprocess execution was routed to
the existing process-group timeout helper. `TemporaryDirectory(delete=False)` retained
all Python-managed scratch on SSD; `PYTHONDONTWRITEBYTECODE=1` prevented probe imports
from writing bytecode into candidate trees. The public shell's own transient compiler
scratch follows its existing cleanup trap. No candidate payload was discarded. The
source-snapshot dry runs keep snapshot verification enabled, with the helper's explicit
`snapshot_root` routed to APDataStore. No smoke waiver is used.

Snapshot hygiene: the full copy contained 90,361 entries and 2,211,884,751 logical bytes,
well beyond its 8,331 mounted-file digest denominator. Compression encountered
`OSError: [Errno 70] Stale NFS file handle`; the partial archive remains retained and is
explicitly **not** reconstruction authority. Cleanup switched to source verification:
43,825 entries matched their retained source counterparts byte-for-byte and were removed
only after the path/hash reconstruction certificate was written. All 46,536 unmatched
entries were AppleDouble or `.DS_Store` metadata; these disappeared with their paired
files, and the final canonical fire-sanitizer pass found zero remaining metadata files.
`source_snapshot_duplicate_certificate.json` pins the source inventory and removal result;
`partial_snapshot_archive.json` records the partial archive's bytes and SHA. No candidate
or historical payload was removed. This restored the storage headroom needed for the
evidence serializer, without retaining a duplicate expanded source tree.

No scorer, paid dispatch, upstream edit, candidate-tree edit, historical-seal edit,
scorer-lane claim or frontier promotion was performed by this arm. CPU replay and full
decode are outside this measurement. All six solver wire-ins are N/A: this apparatus
changes custody metadata and validation, not a sensitivity map, Pareto constraint,
allocator, archive actuator, empirical posterior or optimization probe.

## Landing custody

The serializer could not write Git objects in the shared checkout. Its first fallback
was refused on Vertigo's 40 GiB reserve. The canonical APDataStore fallback succeeded:
`/Volumes/APDataStore/pact/ddm_scg2/serializer/20260909T141739.308263Z-98981/receipts.jsonl`.
The verified code bundle commit is `c1312fc815d7739ef28a1247f93302a08f0f0e78`, bundle
`intended-commit.bundle` in that directory. This commit is retained, not landed on main.
The shared index was not changed. Evidence-document fallback is retained under
`/Volumes/APDataStore/pact/ddm_scg2/evidence_serializer/`.

## NEXT_IF_RESUMED

### ITEM 1 — land the retained code and evidence bundles

- QUEUED-WITH-A-FIRE-ORDER — owner: MAIN; consumer store:
  `.omx/state/canonical_task_status.jsonl`; fire trigger: harvest this arm's final receipt.
  Import and land the serializer's code bundle and evidence bundle, verify their file
  hashes, and record the actual main commits. The code bundle is ready now; the evidence
  bundle receipt is under the directory named above. No score dispatch is requested.

## LIVE-HYPOTHESES

- None required by this bounded custody task. The successor-validation and two-definition
  predictions have been exercised on both historical trees.

## DEAD-ENDS

- INSTANCE, two historical sj1 seals: prose falsifiers alone do not satisfy the structured
  public-smoke contract. They remain immutable historical evidence; use their successors.
- INSTANCE, two historical trees: comparing the seal hash directly to the Modal-projected
  hash as a drift test is invalid. Both match their own historical definitions.
- INSTANCE, first joint smoke: a successful control against the superseded pointer cannot
  anchor a successor against rc2. The fresh run supplies the current control.
- IMPLEMENTATION: patching the original seals or using the missing-smoke waiver would not
  meet this charter's append-only structured-custody requirement.

Live own-vehicle frontier (MAIN's external move, not this arm's measurement):
S 0.13885056455024844 at 181,414 B [contest-CUDA T4 n600], rc2. Sub-0.12 remains unmet.
