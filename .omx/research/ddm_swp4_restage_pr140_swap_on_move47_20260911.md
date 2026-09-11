# ddm_swp4 — PR #140 swap packet re-staged on pointer move 47; publication still refused

`[no-triality] [p0-ledger-ok]` · research_only=true · new_score_claim=false · $0

**The pointer did not move. The packet did.** The staged swap now sits on move 47's bytes
(`d1fab05d…`, 179,359 B, S 0.13654774984742127 `[contest-CUDA T4 n600]`) instead of move 44's three-moves-stale
180,406 B. Strict `--contest-final` compliance is **91/93 PASS, 2 FAIL, rc 1**, with exactly the two known open
items and nothing new. Nothing was pushed, hosted, PR-updated, or published; no dispatch was fired; no receiver
file, sealed tree, live PR tree, or `upstream/` path was written.

Everything lives under `submissions/_staging_move47_pr140_swap/`: `shippable/` is the only candidate release tree
(51 source members + the generated public `report.txt`), `PR_BODY.md` and `SWAP_COMMANDS.md` are sibling drafts, and
`_packet/` holds internal evidence that must never be copied into a public PR.

## Verification

| Obligation | Result and scope |
|---|---|
| Copy identity | 51/51 source members byte-exact against the promoted tree (`_packet/STAGING_MANIFEST.json`); `__pycache__` excluded; `inflate.sh` given the executable bit at staging only |
| Archive | 179,359 B, SHA-256 `d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9` — equal to the charter's pin and to the T4 receipt's `expected_archive_sha256` |
| Retained authority | call `fc-01M29166GTAZC3TA5V8W8Q54E1`, Tesla T4, n600, `gpu_t4_match=true`; no new evaluation was run |
| Recomputed score | **0.13654774984742127**, exact float equality to the receipt's `score_recomputed_from_components` and to the charter |
| Components | d_seg 0.00010345, d_pose 4.59e-06, denominator 37,545,489 B; contributions 0.010345 / 0.00677495387438173 / 0.11942779597303953 |
| Arithmetic order | `rate = B/denominator; S = 100*seg + sqrt(10*pose) + 25*rate` (upstream/evaluate.py:65,92). At move 47's values the reassociated `25*B/denominator` order agrees **bit-for-bit**; move 44's did not. No component differs. |
| Precision | eight-decimal report components plus exact archive bytes; retained worst-case score rounding bound 4.191067489650264e-06 |
| Retained timings | inflate 989.298 s, evaluate 38.652 s (move 44: 1232.419 s / 50.726 s) |
| Source manifest | **49/49** `MANIFEST.sha256` entries verified on the staged tree (`_packet/SOURCE_MANIFEST_CHECK.json`); `README.md` is covered by that manifest, `archive.zip` is not |
| Runtime custody | staged portable runtime tree binds to the auth-eval manifest (`submission_runtime_tree_matches_auth_eval` PASS); expected tree `5170a798…` matches the receipt |
| Dispatch linkage | lane `ddm_hpr1_retrain_control_move46_contest_cuda_20260911`, job `ddm_hpr1_move47_cuda_20260911`, 3 matching rows, latest `completed_contest_cuda_exact_eval_harvested`, binding the exact archive sha **and** runtime-tree sha |
| CPU axis | `fc-01M2929ZX6XCNNX3PNK01A669S`, rc 1 after 8.220 s on move 47's own bytes; refusal receipt sha/bytes equal the adjudication's pin; no CPU metric exists by design |
| Retracted symbols | 43 scanned Python files: **zero** executable identifiers/imports and zero `tc3`/`tc4` module filenames. Ten literal text hits remain (8 `tc3`, 1 `tc4`, 1 `tc4e`) in historical prose and legacy encoded text — the same honest scoping swp3 recorded, not a zero-text claim. |
| Swap procedure | 3 shell blocks and 3 embedded Python heredocs parse; every mutation command is marked NOT RUN and none was executed |
| Boundaries | no tracked repo file modified by this arm (`git status` shows only MAIN's live `.omx/state` rows); the staging tree is untracked, as swp3's is; `/Volumes` was read-only |

## Member-level diff

`_packet/TREE_DIFF.json` holds byte counts and shas for both comparisons.

**vs swp3's move-44 staging** — 52 vs 52 members, 48 identical, **4 changed, 0 added, 0 removed**:
`archive.zip` (179,359 ← 180,406 B), `inflate.py` (2,735 B both — the diff is exactly two lines, `ARCHIVE_SHA256`
and `ARCHIVE_BYTES`), `MANIFEST.sha256` (rebinding that inflate.py hash), and the regenerated `report.txt`.
That is what "the only delta is the pointer" is supposed to look like at the member level, and it independently
corroborates the packet memo's claim that the receiver is unchanged (behavior digest `9f6e7168…`).

**vs the live PR tree** (`submissions/semantic_joint_ctxmix/`, read only) — 52 vs 40 members, 30 identical,
10 changed, 12 present only in the staged tree (`archive.zip`, `report.txt`, and ten runtime/cpr1 modules the live
tree predates). The live tree tracks no `archive.zip`; the asset is hosted separately.

## Strict compliance

Argv, stdout, stderr and all 93 rows are retained in `_packet/COMPLIANCE_COMMAND.json`, `compliance.stdout.txt`,
`compliance.stderr.txt` and `COMPLIANCE.json`. Both open items are unchanged from the reference form and **neither
is closed by assumption**:

| Open check | Cure class / owner | Why it stays open |
|---|---|---|
| `submission_runtime_imports_within_allowlist` | receiver change — operator + MAIN | `runtime/rc3_shared_mixer.py` and `runtime/sm1_semantic_mixer.py` carry `experiments` fallback imports. A receiver edit needs fresh exact evidence and is the operator's decision; this arm does not touch receiver bytes. |
| `hosted_archive_manifest_supplied` | at publish — operator + MAIN | Strict contest-final requires `--hosted-archive-manifest-json`. Nothing is hosted, so the manifest cannot honestly exist yet. |

`_packet/BLOCKERS.json` carries both with owner, disposition and trigger, plus the release debt: the packaged
`README.md` still describes the 180,002-byte ancestor and its score. It is carried byte-exact because
`MANIFEST.sha256` covers it, so refreshing it is a receiver-tree change. `PR_BODY.md` names that staleness in the
body so no stale number can reach a reader silently.

### The r1 diagnostic — derive the guard text, never transcribe it

The first run scored **84/93**: six `contest_cpu_auth_eval_*` checks and `auth_eval_raw_promotion_policy_blockers_absent`
failed with `refusal_guard_does_not_raise_recorded_error`. The cause was mine, not the checker's. I wrote the
refusal receipt's `receiver.error` by reading the source, so it carried the `raise RuntimeError(` wrapper and the
literal quotes; the validator derives that field from the AST, where Python has already folded the two adjacent
string literals into one constant. Deriving the field the same way cured all seven in one edit. `_packet/r1_*`
retains a deterministic replay of the failing input (the original r1 JSON was overwritten before I thought to keep
it — a small custody lapse, named rather than hidden). This is the hv1 negative again at a new surface: never
retype what a receipt or a parser can hand you.

Three dispatch-custody checks that failed for swp3 passed here on the first run, because MAIN's move-47 harvest
poller wrote a terminal claim row binding both the archive sha and the runtime-tree sha.

### Charter premise corrected

The charter states swp3 "sits at 91/93 on MOVE 44". swp3's own memo records **81/93**; cpx3 then hardened the
checker (derive the receiver declaration from the staged tree's real guard; read first-measurement custody from a
completed `candidate_seal.v3`) and re-ran swp3's argv to reach 91/93. The charter inherited the post-cpx3 number
without the lineage. My 91/93 is measured on this packet with cpx3's checker at
`scripts/pre_submission_compliance_check.py` sha `6e47f6c7…`, not carried from any prior claim.

## Public disclosure hygiene

`_packet/PUBLIC_HYGIENE_SCAN.json` lists all 53 scanned public files (`PR_BODY.md` + `shippable/**`) and the ten
patterns applied. **Zero hits** for SSD/volume paths, local home paths, Modal volume paths, `.omx` internal state,
Modal call ids, private/Tailscale IPs, assistant attribution, credentials, and provider names. The checker's own
scan agrees (`public_hygiene.hits == []` over the same 53 files).

One class needs stating rather than suppressing: 117 internal `ddm_*` arm tokens appear inside the receiver source
(module names and comments). The live PR tree contains **the same 117** and already ships
`cpr1/ddm_mp2_semantic_receiver.py` and `runtime/ddm_wc1_advisory_runtime.py`, so the staged tree adds no new
exposure; `PR_BODY.md` contains none. Internal paths and provider details live only in `_packet/` and
`SWAP_COMMANDS.md`, which are never published.

The disclosure sentence is unchanged and names no assistant:

> I used automated research and engineering tools extensively for the work behind this submission.

## Provenance pins

| Input | SHA-256 | Bytes |
|---|---|---:|
| t4 receipt: `/Volumes/VertigoDataTier/pact/ddm_hpr1_fire_move47/run1/MODAL_REMOTE_RESULT.json` | `e49a84cb423492840f020c6956563ad5187be7f6765ae72dd9a17849b9f3b024` | 162,832 |
| cpu receipt: `/Volumes/VertigoDataTier/pact/ddm_hpr1_fire_move47/cpu1/MODAL_REMOTE_RESULT.json` | `843588f9d8050a9b8f072ac603f755f2c19283e55c6e1ea819a038acb1e877be` | 48,034 |
| cpu adjudication: `.omx/research/ddm_hpr1_packet_inputs_20260911/CPU_AXIS_ADJUDICATION.json` | `93e5904c314acb3372f84b1c0dce66cf62ecb88fa018f11dc8a7bc05fb3dcfed` | 1,059 |
| seal: `.omx/research/ddm_hpr1_20260911/SEAL_ddm_hpr1_retrain_control_contest_cuda.json` | `dc112b22f7a62d7c58affbbe66488ec58f3e29acc15f4aefaebb88a1f50a19bf` | 9,071 |
| pointer memo (move 47) | `aa87ded76a987305988e0e524a85586f4d5d1b8a64ce184294ea0b7a9cd60ca1` | 6,230 |
| swp3 memo | `87f1d2ee685d35f11f65809b7c7ef1d3b2671b857714b3147b1f4f038a59a247` | 13,510 |
| cpx3 memo | `f8962504dd1be6ef5a02388c53fa1c4f1a0308d208659bee8393a94bb8ba6089` | 15,129 |
| this charter | `99e9a72ef7293834aee0ebd80e9d143588b1d561d585be63018f4bc4c65bbe17` | 4,055 |
| checker: `scripts/pre_submission_compliance_check.py` | `6e47f6c71a29eef85b144208faad0ef94045b5c40690fffd58ce71722074e343` | 142,782 |

`_packet/INPUT_BINDINGS.json` is the authority for every sha above; nothing in this memo was retyped from memory.

## Boundaries honored

No push, no `gh` invocation, no hosting, no PR edit, no Modal or GPU dispatch, no scorer run, no receiver edit, no
write to any `/Volumes` path, no edit to `upstream/`, the live PR tree, sealed trees, or another arm's directory.
The operator's one-line confirm is the only publish gate and was neither received nor inferred. All six solver
wire-in hooks are N/A for this prepare-only unit: no sensitivity, Pareto, bit-allocator, dispatch, posterior or
probe surface changed.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner operator; consumer `submissions/_staging_move47_pr140_swap/SWAP_COMMANDS.md`;
  trigger the one-line publish confirm: host, fetch back, emit the hosted manifest, require a full strict PASS, then swap.
- QUEUED-WITH-A-FIRE-ORDER; owner operator + MAIN; consumer `_packet/BLOCKERS.json`; trigger the receiver-hygiene
  decision: resolve the `experiments` fallback imports with fresh exact evidence, or record the accepted exception.
- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `shippable/README.md` + `MANIFEST.sha256`; trigger a reviewed
  successor: refresh the README off move 47's numbers and rebind the manifest.
- STANDING; owner the next staging arm; trigger any further pointer move: re-stage rather than repoint this fixed
  move-47 packet, and derive the refusal receiver fields from the staged tree's AST.

composition S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600] (move 47)

## DEAD-ENDS

- INSTANCE: hand-transcribing `receiver.error` from the source text costs seven strict checks. The validator reads
  the folded AST constant; so must the producer.
- INSTANCE: a staged packet cannot honestly satisfy `hosted_archive_manifest_supplied` before hosting. It is not a
  defect to cure, it is the publish gate expressed as a check.
- INSTANCE: refreshing the packaged README is not a free text edit — `MANIFEST.sha256` covers it, so it is a
  receiver-tree change with its own review debt.

<!-- # FORMALIZATION_PENDING: preparation and custody memo only; the retained exact row is recomputed, not re-measured, and no new empirical law or canonical equation is introduced -->
