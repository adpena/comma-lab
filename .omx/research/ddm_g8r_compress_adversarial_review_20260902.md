# DDM G8R generation-8 compressor adversarial review — 2026-09-02

## Verdict

The generation-8 compressor-side surface is hardened and all 37 original public receiver files remain byte-identical to landed commit `7ba53d1b84583305989e7a0c39b9d4fff6f941ef`. The review found real source-pin, resume, environment, embedded-path, compiler-cache, claim-boundary, failure-receipt, and provenance defects; all compressor-side defects were fixed. No receiver file was edited.

The compressor review is **SEALED at 3/3 consecutive post-round-6 clean passes**. The terminal review-owned, retained, resumable bare-tree control executed both complete coder chains and produced two byte-identical 180,002-byte archives with SHA-256 `cbb8d928...`; a final hardened-source resume then revalidated all 39 public manifest rows and all 50 public-plus-embedded source pins. The final strict compliance run is 83 GREEN / 4 RED; therefore this tree is **PREPARED HOLD, NOT READY TO PUBLISH**. The residuals are explicitly owned and do not inherit authority from the older evaluated runtime.

All work in this arm is scorer-free. No Modal job, scorer job, publish action, upstream edit, or frozen generation-7 packet edit occurred.

## RECALL EVIDENCE

The recall searched by content across `.omx/research/`, the APDataStore and Vertigo receipt tiers, `experiments/`, `submissions/`, the canonical research index, `sub015_DAG_*` FEED blocks, design/spec surfaces, and the task-status stores. Queries included `semantic_joint_ctxmix`, `compress.py`, `generation-8`, `bare checkout`, `RC64`, `stale pin`, `Rice-to-CABAC`, `DX2`, `JG2`, `Brotli`, `AppleDouble`, `runtime identity`, the archive SHA `cbb8d928...`, and harness `#1389`. The canonical registry was enumerated with `.venv/bin/python tools/list_canonical_equations.py --json`.

Beyond the charter seeds, recall changed the review in three ways:

- The DAG distinguishes the encoder-bearing RC64 C source (`5c75e2c7...`, 12,222 bytes) from the shipped decoder backend (`05839d14...`, 5,638 bytes). That prevented pinning the wrong native source.
- Live provenance disproved a byte-identical-source reading of the G8C JG2 note. Commit `2c3a2153e4` holds 46,668 bytes at `c93aeda5...`; G8C embedded a deliberate 46,331-byte adaptation at `3a89c2b2...`. G8R now pins both the raw embedded source and the deterministic 46,129-byte public-tree adaptation at `6e2b72e5...`.
- The shipped borrowed-substrate accounting states that pose-basis reorientation was a measured null and was not the shipped move. That removed a false mechanism attribution from README; the shipped contribution is the damped Gauss–Newton coefficient re-solve.
- Generation 7's seven compliance reds were recovered and individually cross-walked below. The bounded task-status-store search did not find a useful G8R owner row; this is only an absence in that searched store, not a claim of global nonexistence.

No fitted model, surrogate, or sampled score estimate was used. The applicable surface was exact byte/hash equality and deterministic replay, so there was no closed-form substitution to approximate: the review used the exact archive/container operators and exact hashes.

## Baseline claim and receipt re-derivation

| Claim | Re-derived result | Axis / boundary | Receipt |
|---|---:|---|---|
| Landed source tree | 40 files, 688,488 bytes, records digest `477f2a3dbf6299ed4cbcc7ffe7ca13becfcaba6c116a5a1dbe8ba8c803ae789e` | exact bytes at actual commit `7ba53d1b84583305989e7a0c39b9d4fff6f941ef` | `initial_landed_tree_rederivation.corrected.json` |
| G8C retained facts | 69/69 distinct file facts pass; 0 missing; 0 mismatched | exact live-custody hashes | `g8c_receipt_fact_rederivation.json` |
| Base archive | 180,456 bytes, `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` | `[macOS-CPU advisory / scorer-free exact byte measurement]` | G8C `bare_proof_v3/RESULT.json` |
| G8C rebuild run 1 | 180,002 bytes, `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, 3,495.107 s | same | G8C `bare_proof_v3/RESULT.json` |
| G8C rebuild run 2 | identical 180,002 bytes and SHA, 3,499.376 s | same | G8C `bare_proof_v3/RESULT.json` |
| Lower-level receiver identity | 600/600 pairs; 3,662,409,600 raw bytes; SHA `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`; 0 differing | `[macOS-CPU advisory / scorer-free exact raw identity]`; not CUDA authority | G8C `runtime_identity_v3/RESULT.json` |
| Final edited source tree | 40 files, 698,446 bytes, records digest `0279694525cd15d03b5f3cd26c1f38ed8af226b14ccc7f3608db5fae9843cfd4`; 39/39 manifest rows pass | exact working-tree bytes | `TREE_DIGEST.final2.json` |

The score `0.14797617125559104` is a prior `[contest-CUDA] Tesla T4 n600` score for the exact archive and evaluated receiver at commit `1c9fbbf58716eb0f26bcdf2a91e3c89d0e4efdde`. It was not re-scored by G8R and is not transferred to the edited public wrapper without current-tree CUDA identity/equivalence proof.

## Recursive findings and counter resets

| Review round | Finding | Severity | Fix or flag | Proof / disposition |
|---|---|---|---|---|
| Finding round 1 | The compressor pinned only three top-level inputs while reading the full receiver/runtime tree; resume could preserve unmanifested staged files. | HIGH | FIXED | `MANIFEST.sha256` is parsed as a 39-row allowlist; all files are hashed and inventories must match exactly. Source tamper, extra-file, hidden-bytecode, and staged-extra controls all refuse. |
| Finding round 1 | Staged copies could retain filesystem metadata and bytecode/AppleDouble state. | HIGH | FIXED | `copyfile` replaces metadata-copy behavior; `sys.dont_write_bytecode` and `PYTHONDONTWRITEBYTECODE=1` are set; `._*`, `__pycache__`, `.pyc`, `.pyo`, `.DS_Store`, symlinks, and special files refuse. |
| Finding round 1 | The base archive could be fetched from a URL, so the build was not a closed local-input replay. | HIGH | FIXED | URL/base-url support and `urllib` are removed. `--base-archive` is mandatory, local, size-pinned, and SHA-pinned. Bare import census finds no network call. |
| Finding round 1 | The native encoder library could be reused by mtime even though its derived `.so` was not content-pinned. | HIGH | FIXED | Native compilation is unconditional, atomic through `.partial`, and the receipt records compiler path, bytes, and SHA. |
| Finding round 1 | The embedded JG2 source retained two lab-specific `/Volumes/...` defaults and its provenance wording implied the wrong source identity. | HIGH | FIXED | Raw embedded SHA `3a89c2b2...` is separately pinned; deterministic adaptation requires explicit `--runtime-root` and `--tokens`, strips defaults, and pins materialized SHA `6e2b72e5...`. Retained provenance diff records the exact transformation. |
| Finding round 1 | Python and declared dependency versions were not enforced even though deterministic replay depends on them. | MEDIUM | FIXED | CPython 3.13.12, NumPy 1.26.4, Torch 2.12.1, and Brotli 1.2.0 are required and recorded. Wrong-Python and missing-compiler compressor controls produce durable REFUSED receipts. |
| Finding round 1 | `--store` defaulted inside the public source tree, allowing payloads/checkpoints to alter the pinned source inventory. | HIGH | FIXED | `--store` is mandatory and must resolve outside the public source tree. Every stage archive, payload, and checkpoint stays on the SSD work tier. |
| Finding round 1 | README wording could associate the old contest-CUDA score with the newly edited public wrapper and did not state CPU/publish boundaries. | HIGH | FIXED | README binds the score to the exact evaluated commit/archive; current-tree T4 proof is explicitly owed; CPU is RECORD-WITH-REASON; publish state is PREPARED HOLD. |
| Finding round 2 | Fallible environment/source checks ran before the top-level RESULT receipt existed, so an early refusal could leave no typed terminal record. | HIGH | FIXED; clean counter reset | After the store itself is resolved and created, the compressor atomically writes `status=RUNNING` before capacity, source, environment, materialization, base-input, or encode checks. Wrong-Python and missing-compiler controls end as durable `status=REFUSED` with the exact error. |
| Finding round 3 | README said the runtime required `linux-nvidia-t4`, while code enforces CUDA and T4 is the prior tested target; CPU wording did not name the actual contest-CPU timeout. | MEDIUM | FIXED; clean counter reset | README now separates enforced CUDA from tested T4 and names the retained 1,800-second contest-CPU timeout. |
| Finding round 3 | README credited pose-basis reorientation, but the shipped accounting proves that reorientation was a measured null and the shipped mechanism was damped Gauss–Newton on existing coefficients. | HIGH / NO-FAKE | FIXED | The null mechanism claim was removed; `455 of 573` now correctly names proposed edits rather than “solved pairs.” |
| Finding round 3 | The source-repository sentence implied live public visibility that this offline review did not establish. | MEDIUM | FIXED | README names the repository and exact evaluated commit but explicitly says visibility of that revision was not re-verified. |
| Finding round 4 | A recalled long object ID for short commit `7ba53d1b84` was wrong. | HIGH provenance | FIXED; clean counter reset | Live `git rev-parse` gives `7ba53d1b84583305989e7a0c39b9d4fff6f941ef`; a new archive/re-hash receipt reproduces the same 40-file digest. The erroneous receipt is superseded, not silently reused. |
| Finding round 5 | `PYTHONDONTWRITEBYTECODE` used `setdefault`, so a caller-supplied empty value could let embedded subprocesses create bytecode state and poison resume. | HIGH | FIXED; clean counter reset | The compressor now forces `PYTHONDONTWRITEBYTECODE=1` in the child environment and separately sets `sys.dont_write_bytecode=True`. |
| Finding round 6 | The draft control table quoted an obsolete expected README hash for the final source-tamper refusal. | MEDIUM provenance | FIXED; clean counter reset | The row now matches `review_terminal_clean2.json`: observed tampered SHA `334359b9...`, expected final README SHA `cbad2dd7...`. |
| Post-round-6 clean 1 | No finding. | — | PASS 1/3 | `review_terminal_clean1c.json`: final import/source/compiler/manifest audit; 39 manifest rows, 50 total source pins, 37/37 receiver files unchanged, no hidden state. |
| Post-round-6 clean 2 | No finding. | — | PASS 2/3 | `review_terminal_clean2c.json`: six fresh executed directions covering unsafe/duplicate manifests, `.DS_Store`, absent manifest, corrupt embedded cache, and stale partial runtime; all refused as designed. |
| Post-round-6 clean 3 | No finding. | — | PASS 3/3 — SEAL | `review_terminal_clean3.json`: two full retained coder chains, final-source resume, 87-row strict compliance, final tree digest, source pins, and hidden-state census all bind. |

The receiver-side residual is `FLAG-CUDA-REPROOF-OWED`, owner MAIN. Editing `inflate.py`, `inflate.sh`, `runtime/`, or `cpr1/` here would invalidate the existing full decode-identity proof, so the arm correctly made no receiver edits.

## Executed controls

| Control | Executed command or mechanism | RC | Exact result |
|---|---|---:|---|
| Final manifest | `(cd submissions/semantic_joint_ctxmix && sha256sum -c MANIFEST.sha256)` | 0 | 39/39 rows OK |
| Bare import closure | `cd bare_source_final && proof_venv313/bin/python -c 'import compress'` | 0 | Imported SSD-tree `compress.py`; `tac` spec is `None` |
| Native compile | `compress.compile_decoder(...)` in clean-pass work store | 0 | retained 33,896-byte `rc64_decoder.so`, SHA-256 `cc4e26a09f68a20d82140b6011ab82296f13922be3586e1a8c3a175f0884be9a` |
| Missing receiver compiler | `CC=__ddm_g8r_missing_cc__ ./inflate.sh ...` | 69 | `requires a C compiler; '__ddm_g8r_missing_cc__' is unavailable (set CC)` |
| Missing receiver Brotli | wrapper run in venv without Brotli | 69 | `requires Brotli==1.2.0; install it before inflation` |
| CUDA unavailable | receiver invoked on macOS CPU | 1 | explicit `requires CUDA inflation on linux-nvidia-t4`; no raw output materialized |
| Wrong compressor Python | system CPython 3.9.6 invokes current tree with durable store | 1 | RESULT is `REFUSED`: `requires CPython 3.13.12; observed CPython 3.9.6` |
| Missing compressor compiler | current compressor with `CC=__ddm_g8r_missing_cc__` | 1 | RESULT is `REFUSED` with exact compiler error |
| Source hash tamper | alter README under copied manifest, then call verifier | 0 harness / refused target | observed `334359b9...` refused against expected `cbad2dd7...` |
| Unmanifested source | add `unmanifested.bin`, then call verifier | 0 harness / refused target | exact-inventory refusal |
| Hidden bytecode | add `__pycache__/sentinel.pyc`, then call verifier | 0 harness / refused target | forbidden-hidden-state refusal |
| Staged-runtime extra | add `unmanifested.bin` to staged runtime, then verify | 0 harness / refused target | exact-inventory refusal |
| Wrong base archive | supply the final archive as the pinned base | 0 harness / refused target | SHA mismatch against `df7fd266...` |
| JG2 required inputs | materialize embedded source and parse its CLI | 0 | `--runtime-root` and `--tokens` required; SHA `6e2b72e5...` |
| G8C live-custody rehash | rehash all distinct paths named by receipts | 0 | 69/69 pass; 0 missing; 0 mismatched |
| Fresh review-owned rebuild | preserved SSD heavy-launch source, local pinned base, retained store, two complete chains; then final SSD source with `--resume` | 0 | heavy execution 7,581.786 s; 10 full encode-stage done rows; two distinct final archives each 180,002 bytes with SHA `cbb8d928...`; final hardened-source revalidation PASS in 0.908 s |
| Strict contest-final compliance | `pre_submission_compliance_check.py --contest-final --strict ...` | 1 | expected HOLD: 83 GREEN / 4 RED; details below |

Control receipts are rooted at `/Volumes/APDataStore/pact/ddm_g8r_compress_adversarial_review/`; bulky source, venv, checkpoints, decoded tokens, compiler outputs, and candidate archives are retained at `/Volumes/VertigoDataTier/pact/ddm_g8r_compress_adversarial_review/`.
`RETENTION_MANIFEST.json` (SHA-256 `a1721a948da532bd78a020b71c3508af3242901877f54a15997512350410d5b4`) binds the exhaustive 702-file, 242,055,207-byte rebuild-store index with records digest `686fb78dbd9120a1a16e4fbb0d364c0ca00ebda825042b06467c901d6b843ac2`; every payload and checkpoint is kept.

## Import and hidden-state closure

The fresh isolated environment is CPython 3.13.12 with exactly NumPy 1.26.4, Torch 2.12.1, and Brotli 1.2.0 as non-stdlib runtime roots; repository package `tac` is absent. The census parsed 44 Python files: 34 public-tree Python files plus 10 materialized embedded sources. External import roots are exactly `brotli`, `numpy`, and `torch`; disallowed imports, parse errors, network calls, lab absolute paths, and hidden-state files are all empty. A real native compiler invocation passed.

The final public source census is 40 files with exactly one Markdown file. `MANIFEST.sha256` has 39 rows because it pins every other public file. There are no `._*`, `__pycache__`, `.pyc`, or `.pyo` entries.

## `DISABLED` adjudication

The sole `"status": "DISABLED"` object in G8C `runtime_identity_v3/RESULT.json` is `$.inflate_report.token_cache`. It is an optional advisory token cache. It is **not** an untested-CUDA leg, and it does not establish that the current public CUDA entrypoint ran.

The correct CUDA adjudication is therefore:

- current-tree public CUDA entrypoint: `ABSENT-UNTESTED-ON-CURRENT-TREE`;
- local evidence: 600/600 lower-level receiver equality on `[macOS-CPU advisory / scorer-free exact raw identity]`;
- disposition: `FLAG-CUDA-REPROOF-OWED`, owner MAIN;
- fire trigger: before publication or current-tree score association, execute the current public entrypoint on 1:1 contest-CUDA Tesla T4 and prove exact raw-output identity, or supply a checker-accepted runtime-equivalence proof.

Receipt: `/Volumes/APDataStore/pact/ddm_g8r_compress_adversarial_review/runtime_identity_adjudication.json`.

## Strict compliance — final per-check rows

Command scope: final edited tree staged on Vertigo, review-rebuilt `cbb8d928...` 180,002-byte archive, existing AFR1 contest-CUDA auth record, `--contest-final --strict`, expected historical runtime hash, terminal dispatch identifiers, README as competitive/public scan text, and no hosted-archive manifest. Receipt: `/Volumes/APDataStore/pact/ddm_g8r_compress_adversarial_review/pre_submission_compliance.gen8.final4.json` (SHA-256 `b05ee6d34dbd072ea8c33e8d849f1629d30e43bea60fa363bedf25a6aaf00c7a`).

| Check | Result |
|---|---|
| `required_file_present:archive.zip` | GREEN |
| `required_file_present:inflate.sh` | GREEN |
| `required_file_present:report.txt` | GREEN |
| `inflate_sh_executable` | GREEN |
| `inflate_sh_uses_canonical_three_arg_contract` | GREEN |
| `inflate_sh_loads_no_scorers_or_eval` | GREEN |
| `archive_exists` | GREEN |
| `zip_member_safe:p` | GREEN |
| `zip_local_header_matches:p` | GREEN |
| `zip_local_header_metadata_matches:p` | GREEN |
| `zip_member_payload_readable:p` | GREEN |
| `archive_zip_readable` | GREEN |
| `zip_no_duplicate_members` | GREEN |
| `zip_expected_single_member` | GREEN |
| `zip_at_most_one_packed_payload_container` | GREEN |
| `expected_archive_sha256_is_well_formed` | GREEN |
| `expected_archive_sha256_matches` | GREEN |
| `expected_archive_size_bytes_matches` | GREEN |
| `auth_eval_exists` | GREEN |
| `auth_eval_json_object` | GREEN |
| `auth_eval_score_parseable` | GREEN |
| `auth_eval_archive_sha_matches` | GREEN |
| `auth_eval_archive_size_matches` | GREEN |
| `auth_eval_schema_metric_consistency` | GREEN |
| `auth_eval_has_components` | GREEN |
| `auth_eval_score_recomputes` | GREEN |
| `auth_eval_strict_formula_score_recorded` | GREEN |
| `auth_eval_selected_axis_matches_submission_gate` | GREEN |
| `auth_eval_score_at_or_below_submission_threshold` | GREEN |
| `auth_eval_t4_equivalent` | GREEN |
| `auth_eval_exact_cuda_stamp` | GREEN |
| `auth_eval_explicit_exact_cuda_stamp` | GREEN |
| `auth_eval_raw_promotion_policy_blockers_absent` | **RED — FLAGGED TO MAIN** |
| `auth_eval_adjudicated_raw_policy_clean` | GREEN |
| `auth_eval_runtime_tree_recorded` | GREEN |
| `auth_eval_runtime_tree_expected_match` | GREEN |
| `contest_cpu_auth_eval_exists` | **RED — RECORD-WITH-REASON** |
| `submission_runtime_inflate_exists` | GREEN |
| `submission_runtime_manifest_computable` | GREEN |
| `submission_runtime_loads_no_scorers_or_eval` | GREEN |
| `submission_runtime_has_no_network_install_or_local_paths` | GREEN |
| `submission_runtime_import_allowlist_parseable` | GREEN |
| `submission_runtime_imports_within_allowlist` | GREEN |
| `submission_runtime_tree_recorded` | GREEN |
| `submission_runtime_tree_matches_auth_eval` | **RED — FLAG-CUDA-REPROOF-OWED** |
| `archive_manifest_exists` | GREEN |
| `archive_manifest_json_object` | GREEN |
| `archive_manifest_sha_matches` | GREEN |
| `archive_manifest_size_matches` | GREEN |
| `archive_manifest_members_present` | GREEN |
| `archive_manifest_member_count_matches` | GREEN |
| `archive_manifest_member_0_name_matches` | GREEN |
| `archive_manifest_member_0_file_size_matches` | GREEN |
| `archive_manifest_member_0_compress_size_matches` | GREEN |
| `archive_manifest_member_0_crc_matches` | GREEN |
| `archive_manifest_member_0_sha256_matches` | GREEN |
| `report_exists` | GREEN |
| `report_mentions_archive_sha256` | GREEN |
| `report_mentions_archive_size_bytes` | GREEN |
| `post_deadline_policy_statement_present` | GREEN |
| `post_deadline_policy_statement_names_mode` | GREEN |
| `post_deadline_policy_statement_not_template` | GREEN |
| `post_deadline_policy_statement_not_negated` | GREEN |
| `post_deadline_policy_statement_has_frontier_context` | GREEN |
| `post_deadline_policy_statement_substantive` | GREEN |
| `hosted_archive_manifest_supplied` | **RED — OPERATOR PUBLISH GATE** |
| `hosted_archive_public_text_has_no_placeholder` | GREEN |
| `public_source_repo_link_present` | GREEN |
| `public_source_pinned_revision_present` | GREEN |
| `public_source_pin_text_has_no_placeholder` | GREEN |
| `public_source_reproducibility_context_present` | GREEN |
| `public_source_reproduce_command_or_sha_binding_present` | GREEN |
| `public_evidence_contest_cuda_label_present` | GREEN |
| `public_evidence_contest_cpu_label_present` | GREEN |
| `public_text_has_no_unresolved_template_placeholders` | GREEN |
| `contest_final_expected_lane_id_supplied` | GREEN |
| `contest_final_expected_job_id_supplied` | GREEN |
| `dispatch_claims_exists` | GREEN |
| `dispatch_claim_terminal_row` | GREEN |
| `dispatch_claim_successful_exact_eval_terminal_row` | GREEN |
| `dispatch_claim_terminal_archive_sha_bound` | GREEN |
| `dispatch_claim_terminal_runtime_tree_sha_bound` | GREEN |
| `dispatch_claim_prior_active_row` | GREEN |
| `public_scan_has_no_private_surface` | GREEN |
| `public_scan_corpus_nonempty` | GREEN |
| `contest_final_selected_axis_auth_score_available` | GREEN |
| `frontier_no_regression_on_submitted_axis` | GREEN |

The checker computes 38 runtime files. The final staged hashes are full tree `f60a8e0d86bbf58b80ad904cb0633cb7bd4c5b7f96a34b44b09671a892f9a7b2` and portable tree `281ef47a76a74a498ab07c08287e283a01d3e3818c3913273d207f8964c29308`; external dependency roots, disallowed imports, forbidden side-effect hits, and import-parse errors are all empty.

### Four RED dispositions

| RED | Plain-language adjudication | Disposition |
|---|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | The raw T4 receipt names policy/rank blockers and predates this adjudicated final compliance receipt. G8R does not rewrite authority custody. | FLAGGED TO MAIN; must be adjudicated in the final promotion packet. |
| `contest_cpu_auth_eval_exists` | No exact AFR1 contest-CPU run exists, and CPU inflation exceeded the 1,800-second budget. | RECORD-WITH-REASON; do not inherit a CPU score. |
| `submission_runtime_tree_matches_auth_eval` | The edited public wrapper/custody tree differs from the historical evaluated runtime. Local lower-level equality is not cross-axis CUDA proof. | `FLAG-CUDA-REPROOF-OWED`; current-tree T4 entrypoint identity/equivalence required before score association. |
| `hosted_archive_manifest_supplied` | No publish or hosting was authorized. | Correct PREPARED-HOLD red; operator approval is the only fire gate. |

### Generation-7 seven-RED crosswalk

| Generation-7 RED | Generation-8 outcome | Disposition |
|---|---|---|
| `auth_eval_raw_promotion_policy_blockers_absent` | RED remains | MAIN adjudicates; no authority laundering. |
| `contest_cpu_auth_eval_exists` | RED remains | RECORD-WITH-REASON. |
| `submission_runtime_tree_matches_auth_eval` | RED remains | Current-tree T4 identity/equivalence fire order. |
| `hosted_archive_manifest_supplied` | RED remains | Operator-only publish gate. |
| `submission_runtime_has_no_network_install_or_local_paths` | GREEN by construction | URL fetch removed; embedded lab defaults removed. |
| `submission_runtime_imports_within_allowlist` | GREEN by construction | Bare import closure and AST census pass. |
| `public_scan_has_no_private_surface` | GREEN by construction | Final public scan covers 43 files with no private-surface hit. |

## What was and was not measured

Measured here:

- exact live hashes for 69/69 G8C receipt facts;
- exact original and final public-tree inventories and manifests;
- bare-environment imports, source parsing, dependency closure, native compilation, and all named refusal controls;
- one review-owned, retained, resumable two-run compressor replay: 10 full encoder-stage executions in 7,581.786 s, followed by final hardened-source manifest/dependency/runtime/archive revalidation through the same retained store;
- strict final compliance over the final edited source tree and exact archive identity.

Not measured here:

- no score and no Seg/Pose components;
- no current-tree contest-CUDA public-entrypoint identity run;
- no contest-CPU authority run;
- no hosted/public archive availability;
- no full decode-identity rerun, because the charter assigns that hours-long proof to the existing receiver receipt and forbids receiver edits.

## SEAL

**SEALED — 3/3 consecutive post-round-6 clean passes.** Clean 1 re-ran the complete source/import/compiler/manifest/receiver audit, clean 2 executed six fresh negative directions, and clean 3 bound the two full coder chains, final hardened-source resume, final source digest, and final strict compliance receipt. The mutable top-level resume receipt's sub-second timings are not presented as rebuild timings; `bare_rebuild_v1/HEAVY_EXECUTION_ADJUDICATION.json` derives the heavy execution from the immutable 10-stage log, original DONE-bound result hash, retained per-stage receipts/checkpoints, and the two distinct final archives.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_g8r_compress_adversarial_review/`, `submissions/semantic_joint_ctxmix/`, and the final promotion packet; fire trigger: after the G8R serializer commit and before any current-tree score association or publication, claim the contest-CUDA lane, run the current public entrypoint on 1:1 Tesla T4, prove exact raw-output identity or checker-accepted runtime equivalence, re-run strict compliance, adjudicate the raw-policy blocker, preserve CPU as RECORD-WITH-REASON, and publish only after explicit operator approval supplies the hosted-archive manifest.

## LIVE-HYPOTHESES

- The current public entrypoint will reproduce the historical T4 raw output because no lower-level receiver file changed and the 600/600 macOS lower-level identity passed; this remains a hypothesis until the wrapper itself runs on T4.

## DEAD-ENDS

- **INSTANCE closed:** URL/base-archive fallback is incompatible with a closed, reproducible public compressor; local pinned base input is now mandatory.
- **INSTANCE closed:** partial top-level pinning cannot protect a full runtime-tree consumer; exact manifest inventory and hashes are now mandatory.
- **INSTANCE closed:** mtime-based reuse of an unpinned native library is not deterministic custody; the library is rebuilt atomically every run.
- **INSTANCE closed:** G8C's embedded JG2 adaptation is not byte-identical to commit `2c3a2153e4`; raw, adapted, and materialized identities are now separate.
- **INSTANCE closed:** `$.inflate_report.token_cache = DISABLED` is not the CUDA leg and cannot close current-tree CUDA proof.
- **INSTANCE closed:** ExFAT APDataStore is unsuitable as the clean execution tree because ordinary writes create `._*`; execution stays on Vertigo/APFS and APDataStore is receipt custody only.
- **INSTANCE closed:** a detached shell without a persistent supervisor was reaped before producing payload; the retained rebuild uses the required supervisor, pidfile, durable log, stage checkpoints, and resume store.

Own-vehicle frontier: **UNMOVED** — AFR1 exact `[contest-CUDA T4 n600]` `S=0.14797617125559104` at `180,002 B`, archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.
