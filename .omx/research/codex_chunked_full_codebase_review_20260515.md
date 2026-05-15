# Codex chunked full-codebase adversarial review ledger 2026-05-15

**Operator directive:** *"we need a /codex:adversarial-review against the whole current state of the codebase chunked to avoid timeout"*

**Completed:** 10/10 chunks (3 missing chunks re-run by `HARVEST-CHUNKED-CODEX-MISSING-AND-AGGREGATE-20260515` 2026-05-15).

## Per-chunk verdict + finding-count summary

| Chunk | Name | Verdict | H | M | L | Total | Top finding |
|---|---|---|---|---|---|---|---|
| 1 | core_library | needs-attention | 2 | 2 | 0 | 4 | Catalog #242 validates only the example registry, not the trainer contracts |
| 2 | preflight_infra | needs-attention | 2 | 1 | 0 | 3 | Catalog drift coverage misses lambda-wired strict gates and unrowed references |
| 3 | substrate_meta_layer | needs-attention | 2 | 2 | 0 | 4 | DUPLICATE of chunk 1 (codex companion ran adversarial-review against working tree diff WITHOUT per-chunk path scoping) |
| 4 | operator_authorize_dispatch | needs-attention | 2 | 2 | 0 | 4 | Risk scorer never gates dispatch and uses an empty verdict panel |
| 5 | modal_lightning_vastai | needs-attention | 3 | 2 | 0 | 5 | Canonical Modal ledger is best-effort after paid spawn |
| 6 | rudin_daubechies_autopilot | needs-attention | 3 | 3 | 0 | 6 | Preflight REFUSE cannot block dispatch |
| 7 | substrates_ara_composition | needs-attention | 2 | 1 | 0 | 3 | Byte-mutation verifier cannot produce the new proof records |
| 8 | commit_machinery_wave_d | needs-attention | 4 | 2 | 0 | 6 | Expected-hash protection can be partially satisfied |
| 9 | claude_md_discipline_burden | needs-attention | 2 | 1 | 0 | 3 | Strict memory-backed gate passes vacuously off the original machine |
| 10 | submissions_inflate_runtime | needs-attention | 1 | 0 | 0 | 1 | Zero-artifact partial harvests are closed as successful completed jobs |

**Severity totals across 9 unique chunks (excluding chunk 3 dup):** HIGH=21 / MEDIUM=14 / LOW=0 / INFO=0 = **35 findings**.

All chunks `needs-attention`. Codex's `needs-attention` is its standard adversarial verdict; the actual operator decision is per-finding. Per CLAUDE.md "KILL is the LAST RESORT" non-negotiable: every finding receives DEFERRED-pending-fix verdict with reactivation criteria. Zero kills.

## Per-chunk verbatim codex output

### Chunk 1: core_library (`review-mp7f2a26-y972k4`)

**Verdict:** `needs-attention`

**Summary:** No-ship: the META and dispatch gates are still declaration-based and under-scoped; they can greenlight unvalidated substrate contracts, known L2 integration gaps, and unproven optimization/feature-consumption claims.

**Findings:**

#### 1. [HIGH] Catalog #242 validates only the example registry, not the trainer contracts

- **File:** `src/tac/preflight.py` lines 56580-56589; confidence 0.94
- **Body:** Catalog #242 claims to validate every file using @register_substrate, but the implementation only imports tac.substrate_registry.example_template before calling validate_all_registered(). Because registration is import-side-effect based, the real trainer modules are not loaded; a read-only probe left the registry with only example_template while rg shows 27 decorated train_substrate_*.py files. Catalog #241 is also warn-only and currently reports 5 trainers with neither decorator nor legacy tag. Impact: malformed, missing, or stale 36-field contracts in real trainers will not block strict preflight or downstream recipe/driver generation.
- **Recommendation:** Have #242 enumerate all experiments/train_substrate_*.py files and validate each contract via safe isolated import or static extraction; fail on import/validation errors. Make #241 strict for dispatchable trainers or require explicit non-dispatchable legacy tags.

#### 2. [HIGH] Distinguishing-feature gate accepts truthy metadata instead of proof

- **File:** `src/tac/preflight.py` lines 58375-58408; confidence 0.92
- **Body:** _check_272_lane_has_required_fields treats byte_mutation_smoke_passes=True, any non-empty dict, or any string path as sufficient. It never opens the verifier JSON, checks that the path exists, confirms mutated bytes match distinguishing_bytes_path, verifies inflate_consumer_function ran, or requires a raw-frame/scorer delta. The gate is also wired warn-only in preflight, and the live check reports 6 L2+ violations. Impact: false positives are easy: bytes can flow through headers, padding, error paths, or registry assertions while the distinguishing feature produces no meaningful output change.
- **Recommendation:** Parse the canonical proof artifact and require per-section mutation of declared bytes, successful source/mutated inflate, changed raw output or explicit semantic delta, and linkage to the named consumer function. Run strict for L2+ promotion and dispatch readiness.

#### 3. [MEDIUM] Recipe-vs-trainer guard only catches the survived NotImplementedError pattern

- **File:** `src/tac/preflight.py` lines 54653-54682; confidence 0.78
- **Body:** Catalog #240 classifies the trainer as complete whenever _full_main is absent or does not contain a literal raise NotImplementedError in the first 3000 characters. That catches the documented Z4/Z5 crash signature, but misses other recipe-vs-trainer divergences: recipes pointing at the wrong CLI/profile, full paths silently taking smoke/research-only branches, implemented _full_main bodies that never emit an archive/auth-eval, or alternate entrypoints with no _full_main. This creates survivorship bias: crashing stubs are counted, silent no-op or smoke-only misroutes can pass as complete.
- **Recommendation:** Require positive dispatch-chain evidence for contest recipes: parse the recipe/trainer CLI contract, prove the full path is selected, and require a dry-run/local pre-deploy artifact showing archive and auth-eval handoff. Treat missing _full_main as unknown unless an explicit alternate entrypoint contract is declared.

#### 4. [MEDIUM] Tier-1 optimization protocol is token grep, not per-substrate speed evidence

- **File:** `tools/canonical_dispatch_optimization_protocol.py` lines 247-269; confidence 0.82
- **Body:** _verify_tier1 marks autocast, TF32, torch.compile, no_grad, and canonical scorer loss as present if any configured token appears anywhere in the trainer text. A declared CLI flag, waiver token, or unrelated eval helper can satisfy the gate without the hot training loop actually using the primitive. This does not support claims that the T1 Ballé speed pattern transfers to each substrate, and can authorize paid dispatches under a false timeout-prevention assumption.
- **Recommendation:** Replace token presence with executable or AST checks tied to hot-loop callsites, and require a small timing-smoke artifact per substrate/flag bundle before treating a Tier-1 primitive as operationally effective.

**Next steps:**

- Block shipment until #241/#242 validate every real trainer contract, not just the example registry.
- Promote #272 from truthy declaration to parsed proof with raw-output or scorer-delta evidence.
- Add positive dispatch-chain and timing-smoke evidence before using these gates to authorize paid substrate dispatches.

---

### Chunk 2: preflight_infra (`review-mp7ezp82-v24861`)

**Verdict:** `needs-attention`

**Summary:** No-ship for the requested preflight surface: the meta-catalog layer is not actually self-covering, and warn-only findings can be cached out of sight.

**Findings:**

#### 1. [HIGH] Catalog drift coverage misses lambda-wired strict gates and unrowed references

- **File:** `src/tac/preflight.py` lines 45836-45860; confidence 0.93
- **Body:** Catalog #176 claims to require CLAUDE.md rows for strict preflight callsites, but its collector only matches lines that begin with a direct `check_...(` call. It misses `_parallel.run(..., lambda: check_foo(strict=True))` wiring; I found 63 strict lambda callsites, including 39 with no CLAUDE row, such as `check_no_comment_only_contracts` at line 3606. This is the same blind spot class as the referenced Catalog #216: `check_commit_serializer_post_stage_hash` is mentioned as a sister gate, but there is no `^216.` CLAUDE row and no callable in `src/tac/preflight.py` for #185 to verify. Impact: operators get false confidence that the catalog ledger covers all strict gates.
- **Recommendation:** Replace the regex collector with an AST-backed preflight graph validator that discovers direct calls, lambda calls inside `_parallel.run`/`submit`, and `preflight_*` functions. Add a catalog-reference resolver that fails when any `Catalog #N` / backticked `check_*` reference lacks a numbered row plus either a callable preflight gate or an explicit external-gate contract.

#### 2. [HIGH] Clean-cache path hides warn-only violations after the first run

- **File:** `src/tac/preflight.py` lines 408-436; confidence 0.9
- **Body:** Warn-only checks can return findings, `_record_result` only increments `advisory_count`, and `_store_preflight_codebase_clean_cache` still records a clean cache row with that advisory count. The cache-hit predicate ignores `advisory_count`, so the next identical invocation sets `check_codebase = False` and skips the warn-only gates entirely. That turns warn-only into silent-only, directly undermining strict-flip discipline and making backfill pressure disappear.
- **Recommendation:** Do not store a clean cache row when `advisory_count > 0`, or include an advisory fingerprint/list in the cache and replay it on cache hits. For strict-flip candidates, treat any nonzero advisory count as a cache miss until the gate is strict or retired.

#### 3. [MEDIUM] Catalog #185 fails open when a gate signature drifts

- **File:** `src/tac/preflight.py` lines 47025-47030; confidence 0.86
- **Body:** The live-count-zero meta gate catches `TypeError` from `fn(strict=False, verbose=False)` and silently moves the gate into `skipped`. A future refactor that adds required args or changes the signature would stop #185 from executing the underlying gate while still allowing the overall check to return clean if nothing else fails. That is exactly a catalog-row drift class, but the meta gate treats it as non-blocking.
- **Recommendation:** Fail closed on `TypeError` unless the function is in an explicit skip list with a durable reason. For gates needing kwargs, require a small probe adapter registered in the catalog metadata rather than implicit skipping.

**Next steps:**

- Consolidate `#118/#159/#176/#185` into one AST-backed catalog graph validator.
- Consolidate commit/subagent integrity gates such as `#117/#157/#174/#186/#206/#230/#234/#235/#248` behind one rule-table driven checker.
- Move substrate recipe/trainer optimization overlap (`#170/#171/#172/#178/#179/#180/#181/#182/#215/#226/#270`) behind the canonical dispatch optimization protocol, with leaf gates kept as focused unit-tested helpers.

---

### Chunk 4: operator_authorize_dispatch (`review-mp7f5zrb-q7gbkn`)

**Verdict:** `needs-attention`

**Summary:** No ship: the dispatch gate adds visible ceremony, but the highest-risk paths still proceed or can be scripted around.

**Findings:**

#### 1. [HIGH] Risk scorer never gates dispatch and uses an empty verdict panel

- **File:** `tools/operator_authorize.py` lines 1521-1551; confidence 0.94
- **Body:** A REFUSE-class preflight risk can still dispatch. The helper builds GateVerdictPanel(verdicts={}) instead of collecting actual staged-file gate results, treats import/scoring failures as warnings, and only prints WARN/REFUSE rationale without aborting. Inference from the scorer contract: an empty panel has no violated gates, so cold-start prediction is effectively OK. This turns Catalog #273-#278 into false assurance before paid dispatch.
- **Recommendation:** Feed real preflight gate verdicts into the panel and raise on REFUSE unless an explicit, audited bypass is present; make scorer import/evaluation failure fail closed for native dispatch.

#### 2. [HIGH] Paired-env clean bypass is automatically scripted

- **File:** `tools/run_modal_smoke_before_full.py` lines 748-751; confidence 0.91
- **Body:** The paired-env discipline does not prevent accidental bypass when a wrapper sets both variables automatically. On any dirty tree, the smoke wrapper sets OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 and OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1 itself, with no independent sentinel-clean proof at this call site. That makes the second env var ceremony rather than attestation and can dispatch smoke jobs from unintended dirty code.
- **Recommendation:** Do not auto-set the attestation variable. Either verify and record the exact sentinel manifest/hash before setting it, or require an operator-supplied bypass reason/nonce that wrappers cannot synthesize silently.

#### 3. [MEDIUM] Local harness executes trainer code before dispatch

- **File:** `tools/local_pre_deploy_check.py` lines 264-359; confidence 0.86
- **Body:** The 30-second harness is not just py_compile/static checking: it imports the trainer module and may call _build_archive_zip. Trainer imports can initialize heavy optional dependencies, perform module-level side effects, hang, or fail on local-only dependency skew even when the remote dispatch would be valid. Because operator_authorize runs this in strict mode, these false positives can train operators to bypass the harness.
- **Recommendation:** Make the default harness import-free and AST/static. If dynamic checks are required, run them in a subprocess with a hard timeout, clean environment, and explicit safe-entrypoint contract.

#### 4. [MEDIUM] Mandatory pre-deploy harness fails open when missing

- **File:** `tools/operator_authorize.py` lines 438-445; confidence 0.88
- **Body:** operator_authorize says every native dispatch must pass the local harness, but if tools/local_pre_deploy_check.py is absent it only warns and returns. A bad merge, partial checkout, or path drift can silently remove the guard while still allowing GPU dispatch, which defeats the fail-closed claim.
- **Recommendation:** Raise SystemExit when the harness is missing for native dispatch, unless the same audited bypass path used for intentional skips is active.

**Next steps:**

- Wire the risk scorer to real gate verdicts and enforce REFUSE before claim creation.
- Replace automatic bypass attestation with verified sentinel evidence or an operator-supplied token.
- Make local_pre_deploy_check static by default; isolate dynamic checks with timeout and no side effects.
- Change missing-harness behavior from warning to fail-closed.

---

### Chunk 6: rudin_daubechies_autopilot (`review-mp7fbln2-5fc65r`)

**Verdict:** `needs-attention`

**Summary:** No-ship: the new autopilot/preflight risk layer is mostly uncalibrated, and the dispatch integration does not actually enforce the risks it claims to surface.

**Findings:**

#### 1. [HIGH] Preflight REFUSE cannot block dispatch

- **File:** `agent_worktree_copy/tools/operator_authorize.py` lines 1521-1551; confidence 0.93
- **Body:** `_consult_preflight_risk_scorer` builds a clean `GateVerdictPanel(verdicts={})`, uses a fresh default scorer with no persisted anchors, and then only logs WARN/REFUSE. In practice this makes native dispatch see cold-start risk 0/OK, and even a real REFUSE would not abort. Likely impact: paid GPU jobs can proceed through preflight conditions this feature claims should stop, burning race budget and hiding failures behind advisory logs.
- **Recommendation:** Populate the panel from actual staged-file/preflight gate results, load the persisted scorer store, and raise before lane claim/dispatch on REFUSE unless an explicit bypass flag and ledger entry are present.

#### 2. [HIGH] Autopilot risk field is dead in ranking

- **File:** `agent_worktree_copy/tools/cathedral_autopilot_autonomous_loop.py` lines 820-880; confidence 0.88
- **Body:** `predicted_dispatch_risk` is documented as the signal that should defer high-risk candidates, but `rank_candidates` ignores it entirely. The EIG and score-delta sort keys only apply MDL, posterior, class-shift, and composition adjustments. Likely impact: a candidate with `predicted_dispatch_risk >= 50` can still be ranked first and recommended for dispatch.
- **Recommendation:** Make risk an explicit sort/filter input: refuse or down-rank candidates above the refusal threshold, record the risk rationale in halt events, and add a regression test proving a high-risk candidate cannot outrank an otherwise equal low-risk candidate.

#### 3. [MEDIUM] Integer SLIM can replace ranking without accuracy proof

- **File:** `agent_worktree_copy/src/tac/autopilot_rudin_daubechies/slim_ranker.py` lines 398-489; confidence 0.78
- **Body:** The SLIM fitter commits the greedy integer-constrained model based only on in-sample SSE. There is no top-k recall, rank-correlation, holdout, or comparison against the existing ranker/continuous model before its predictions drive dispatch ordering. Inference: the interpretability constraint can silently underfit enough to miss the optimal dispatch while still producing an auditable-looking explanation.
- **Recommendation:** Gate SLIM use behind empirical rank-quality thresholds on held-out anchors, such as top-k recall and Spearman/Kendall versus contest-CUDA outcomes or the prior ranker; fall back when below threshold.

#### 4. [MEDIUM] Rashomon disagreement treats low evidence as consensus

- **File:** `agent_worktree_copy/src/tac/autopilot_rudin_daubechies/rashomon_ensemble.py` lines 146-160; confidence 0.82
- **Body:** The ensemble reports raw stddev across K members, but K is fixed at 8 and there is no minimum anchor diversity or calibration. With empty or duplicate/single-anchor bootstrap samples, all members can predict identically and emit zero disagreement, converting lack of evidence into apparent consensus.
- **Recommendation:** Report an `insufficient_anchor_diversity` state until enough distinct anchors exist, run K-sensitivity calibration, and calibrate disagreement thresholds against empirical miss rates before using disagreement as a dispatch signal.

#### 5. [HIGH] Compressive landscape is not sparse recovery

- **File:** `agent_worktree_copy/src/tac/autopilot_rudin_daubechies/compressive_landscape.py` lines 197-229; confidence 0.9
- **Body:** The code claims L1 recovery in a wavelet basis, but `_haar_l1_reconstruct` is inverse-distance interpolation over cell index. There is no sparsity measurement, no residual/holdout validation, and uncertainty can collapse when observed score variance is small. Likely impact: unmeasured landscape cells can look confidently ranked despite the Daubechies-DeVore sparsity assumptions never being checked.
- **Recommendation:** Either implement real sparse recovery with explicit sparsity and reconstruction-error diagnostics, or downgrade this to heuristic interpolation and prevent it from producing high-confidence dispatch rankings without holdout validation.

#### 6. [MEDIUM] GOSDT routing bottlenecks on any fired rule

- **File:** `agent_worktree_copy/src/tac/preflight_rudin_daubechies/gosdt_dispatch_router.py` lines 179-207; confidence 0.86
- **Body:** The router derives risk only from `fired_rule_count`. Because one fired rule gives `estimated_risk_high == 25` and the OK check is strict `< 25`, even a single low-severity fired rule falls through to `REQUEST_OPERATOR_REVIEW`. In race mode this can serialize dispatch on operator review for routine warnings.
- **Recommendation:** Use calibrated tier-specific SLIM/Rashomon risk rather than raw fired count, and define race-mode actions for low-risk warnings so only genuinely ambiguous/high-risk paths require operator review.

**Next steps:**

- Block shipping until risk consultation is wired to real preflight verdicts and enforced before dispatch.
- Add calibration tests for SLIM rank quality, Rashomon disagreement, and compressive-landscape uncertainty before using them in race-mode ranking.

---

### Chunk 7: substrates_ara_composition (`review-mp7feoxg-f3m9k4`)

**Verdict:** `needs-attention`

**Summary:** No-ship: the proof surface is still declaration-heavy, the byte-mutation verifier appears broken on its main result path, and cross-substrate composition is still ranked from assumed additivity rather than empirical HStack/VStack anchors.

**Findings:**

#### 1. [HIGH] Byte-mutation verifier cannot produce the new proof records

- **File:** `tools/verify_distinguishing_feature_byte_mutation.py` lines 429-486; confidence 0.93
- **Body:** The verifier now defines required SectionResult fields for target_basis, member, offset, and length, but the main mutation loop still instantiates SectionResult without those fields for both empty and non-empty sections. That means the Catalog #272 byte-mutation smoke path raises TypeError before it can emit a valid proof artifact. The advertised byte-range support is also unreachable from the public CLI, so the new MutationTarget plumbing is effectively dead code.
- **Recommendation:** Thread MutationTarget objects through the verifier and CLI, add a byte-range argument, populate the new SectionResult fields on every return path, and add a regression test that runs the verifier end-to-end against an archive section and a byte range.

#### 2. [HIGH] Catalog #220 still accepts operational self-attestation

- **File:** `src/tac/preflight.py` lines 54458-54466; confidence 0.88
- **Body:** The preflight gate collects free-text lane evidence and treats substring tokens such as operational status or runtime overlay consumption as sufficient proof. It does not load a verifier artifact, check archive/runtime hashes, or run the byte-mutation smoke tool before allowing the byte-addition mechanism through. A lane can therefore satisfy the gate by writing the right words into registry evidence, which is exactly the self-attestation failure mode the operational-mechanism declaration is supposed to prevent.
- **Recommendation:** Replace token-only acceptance with a structured required proof reference, then validate the referenced byte-mutation smoke JSON, archive SHA, runtime SHA, command, return code, and mutated-byte effect before treating a lane as operational.

#### 3. [MEDIUM] Cross-substrate dispatch ranking extrapolates additive composition

- **File:** `src/tac/optimization/autopilot_dispatch_ranking.py` lines 200-239; confidence 0.78
- **Body:** The orthogonal pair builder emits joint HStack/VStack-style candidates by summing pair EIG and calling predicted_composite_delta. The underlying composition matrix uses taxonomy-derived alpha values, not per-pair empirical composition anchors. Although the candidate is not marked exact-eval ready, it can still steer automated dispatch ranking toward paid or scarce experiments based on single-substrate assumptions rather than measured cross-substrate additivity.
- **Recommendation:** Require an empirical composition anchor for each substrate pair before it can enter dispatch ranking, or attach a blocking reason such as cross_substrate_anchor_missing and keep it out of automated top-k dispatch until a real paired smoke/eval artifact exists.

**Next steps:**

- Fix and test the byte-mutation verifier first; it is the evidence generator other gates need.
- Harden Catalog #220 and #272 to consume structured proof artifacts instead of registry text tokens.
- Add composition-anchor metadata and exclude unanchored substrate pairs from dispatch ranking.

---

### Chunk 8: commit_machinery_wave_d (`review-mp7fh9pn-1ie8yr`)

**Verdict:** `needs-attention`

**Summary:** No-ship: the commit/catalog/checkpoint protection layer still has bypasses, crash-corruption paths, and catalog drift that would make future self-protection claims unreliable.

**Findings:**

#### 1. [HIGH] Expected-hash protection can be partially satisfied

- **File:** `tools/subagent_commit_serializer.py` lines 420-428; confidence 0.92
- **Body:** The staged-content guard only checks paths present in `expected`, not every path passed via `--files`. A caller can include one `--expected-content-sha256` for a safe file while committing unprotected high-risk files in the same invocation; Catalog #174's source scan only proves the flag appears somewhere. That leaves the pre-pre-lock sibling-edit absorption class open even when the command appears compliant.
- **Recommendation:** Require normalized `set(expected_content_shas) == set(files)` for all non-`--no-stage` commits, and make Catalog #174 enforce one expected hash per committed file. Log the covered file set on success.

#### 2. [HIGH] Catalog counter can reset after a crash mid-write

- **File:** `tools/claim_catalog_number.py` lines 149-157; confidence 0.91
- **Body:** `claim_one()` truncates `next_catalog_number.txt` and then rewrites it in place; if the process dies after truncate but before the write/fsync completes, the fcntl lock is released and the next claimant reads an empty file. The code treats empty as `DEFAULT_INITIAL_VALUE`, which can reissue old catalog numbers. This is exactly the high-cost failure mode for contending canonical helpers plus crash.
- **Recommendation:** Fail closed on empty/corrupt state, write via temp file + fsync + atomic rename or an append-only checksummed claim journal, and keep claim/commit under one transaction lock or persist an auditable pending-claim record.

#### 3. [MEDIUM] Checkpoint corruption is silently hidden

- **File:** `tools/subagent_checkpoint.py` lines 238-241; confidence 0.87
- **Body:** `read_checkpoints()` drops malformed JSONL rows with `continue`. If the checkpoint writer crashes after a partial append, successors get no error and may resume from stale state or believe no checkpoint exists. Since this tool exists specifically for crash recovery, silently hiding the damaged record defeats the protection layer.
- **Recommendation:** Fail closed on malformed non-empty rows, quarantine/repair only through an explicit recovery path, and add sequence/checksum metadata so readers can distinguish a clean prefix from a torn latest record.

#### 4. [MEDIUM] WAVE-D proposes a log gate the log cannot support

- **File:** `.omx/research/commit_swap_incident_2c957c31e_forensic_analysis_20260515.md` lines 67-76; confidence 0.95
- **Body:** The proposed strict gate relies on detecting commits with absent `expected_content_sha_diffs` or missing `expected_content_sha256` in `commit-serializer.log`. Successful serializer records do not currently log whether expected hashes were passed or which files they covered, so the proposed detector would either flag compliant commits or miss the drop-flag pattern.
- **Recommendation:** First extend successful serializer log records with `expected_content_sha256_present`, covered file list, and coverage count; then build the retrospective gate against that explicit signal.

#### 5. [HIGH] Catalog drift checker misses current phantom rows

- **File:** `src/tac/preflight.py` lines 46923-46931; confidence 0.9
- **Body:** Catalog #185 only inspects entries matching a narrow live-count phrase list. A parse of the current checkout finds six CLAUDE catalog rows (#273-#278) whose `check_*` functions are absent from root `src/tac/preflight.py`, while #185 returns clean because those rows say `live count at landing: 0`, a wording not matched here. #216 is another phantom reference, but as a cross-reference rather than a numbered row.
- **Recommendation:** Add an unconditional `CLAUDE.md catalog row -> callable gate` check independent of live-count wording, expand the live-count parser, and either land the missing #273-#278/#216 callables or remove/mark the rows as non-callable references.

#### 6. [HIGH] Worktree isolation only defers edit collisions

- **File:** `tools/subagent_commit_serializer.py` lines 91-95; confidence 0.84
- **Body:** The serializer derives `REPO_ROOT` from its own file path, so a sibling worktree uses its own `.omx/state/.commit-lock` and `.omx/state/commit-serializer.log`. That means same-machine worktrees do not contend on the same lock, and main-worktree gates that inspect the main log cannot see sibling serializer commits. This prevents shared-working-tree overwrites but moves overlapping edits to merge time, where attribution and catalog/function atomicity can still split.
- **Recommendation:** Move the lock/log to a shared git-common-dir or workspace-level coordination path, and require a merge helper that imports sibling serializer logs and rejects overlapping path ownership before landing.

**Next steps:**

- Make expected-hash coverage mandatory per file before relying on Catalog #157/#174/#216 claims.
- Harden state writes with atomic persistence and fail-closed readers before further fan-out.
- Add a real catalog-row-to-callable drift gate, then backfill #216 and the six current phantom rows.

---

### Chunk 9: claude_md_discipline_burden (`review-mp7fjuiw-or9b0q`)

**Verdict:** `needs-attention`

**Summary:** No-ship: the change still relies on oversized docs and private auto-memory as the control plane, while the new gates mostly detect drift after the fact or accept proof artifacts that can go stale.

**Findings:**

#### 1. [HIGH] Strict memory-backed gate passes vacuously off the original machine

- **File:** `src/tac/preflight.py` lines 35316-35340; confidence 0.93
- **Body:** The landing-memo gate resolves feedback_*.md from PACT_MEMORY_DIR or ~/.claude/projects/<repo>/memory, then returns a clean result on GitHub Actions when that external directory is missing. That means the rules claimed to be enforced from the auto-memory corpus are not reproducible for a fresh operator, CI runner, or cloned checkout. Impact: a missing or stale 800+ file feedback corpus can silently remove the solver-wire-in gate while preflight still appears green.
- **Recommendation:** Make the canonical feedback corpus or a generated hash/index repo-relative and committed, and fail closed in strict mode when it is missing unless an explicit recorded waiver is passed.

#### 2. [HIGH] Catalog #272 accepts stale proof artifacts as current lane evidence

- **File:** `src/tac/preflight.py` lines 58446-58476; confidence 0.88
- **Body:** The new artifact-bound check reads the path from byte_mutation_smoke_passes and validates only the proof schema plus target names. It does not bind the proof to the lane's current archive SHA, runtime tree, inflate.sh hash, or candidate artifact path. A lane can change its packet/runtime after producing a passing proof, keep the old proof path, and still satisfy this validator. This directly leaves catalog-row drift between CLAUDE.md contract text and empirical state.
- **Recommendation:** Require lanes to declare the candidate archive/runtime identity and reject proofs whose archive_sha256, inflate_sh_sha256, runtime tree SHA, and target section hashes do not match the current lane/candidate.

#### 3. [MEDIUM] Mandatory full-doc pre-read is not an operable control plane

- **File:** `CLAUDE.md` lines 280-290; confidence 0.82
- **Body:** CLAUDE.md says the docs themselves are the orchestration layer and requires every subagent to read CLAUDE.md and AGENTS.md and honor every NON-NEGOTIABLE marker. The measured docs are 77,289 words with 40 NON-NEGOTIABLE headings, before the private feedback corpus is considered. This shifts correctness onto impossible human/agent recall instead of a bounded active contract, making fresh-agent and fresh-operator failures hard to detect.
- **Recommendation:** Replace full-doc mandatory pre-read with a generated short active contract: prioritized obligations, owning gate, file-line source, freshness/hash metadata, and a design-time check that rejects new non-negotiables without that index entry.

**Next steps:**

- Promote memory-backed rule sources into committed repo-relative ledgers or manifests.
- Bind every runtime proof to current archive/runtime custody, not just to a path and schema.
- Split CLAUDE.md into active machine-checkable contract plus historical appendix before adding more catalog rows.

---

### Chunk 3: substrate_meta_layer (re-run via `HARVEST-CHUNKED-CODEX-MISSING-AND-AGGREGATE-20260515`)

**Verdict:** `needs-attention` (DUPLICATE of chunk 1).

**Summary:** No-ship: the META and dispatch gates are still declaration-based and under-scoped; they can greenlight unvalidated substrate contracts, known L2 integration gaps, and unproven optimization/feature-consumption claims.

**Note:** The codex companion `adversarial-review` invocation defaults to reviewing the working-tree diff, not a path-scoped subset. Chunks 1 and 3 produced byte-identical final assistant messages because both were dispatched against the same working-tree diff. Per CLAUDE.md "Forbidden premature KILL": chunk 3 is preserved in the registry for forensic continuity. If codex companion gains per-chunk `--paths` scoping in a future release, chunks 1 + 3 should be re-run with explicit path filters and the verdicts compared.

Findings duplicate chunk 1 verbatim (H-1 + H-2 + recipe-vs-trainer + Tier-1 token grep). See chunk 1 above.

---

### Chunk 5: modal_lightning_vastai (re-run via `HARVEST-CHUNKED-CODEX-MISSING-AND-AGGREGATE-20260515`)

**Verdict:** `needs-attention`

**Summary:** No target-path working-tree hunks are present, but the live provider/runtime surfaces still have ship-blocking audit and concurrency gaps.

**Findings:**

#### 1. [HIGH] Canonical Modal ledger is best-effort after paid spawn

- **File:** `experiments/modal_train_lane.py` lines 1393-1419
- **Body:** The dispatch path launches the Modal job, writes per-dispatch sentinels, then attempts the canonical call_id ledger append inside a broad try/except that only prints a warning. If the ledger is corrupt, locked, or times out, the paid job still exists and the CLI can continue without the single queryable Catalog #245 index. That recreates the orphan call_id failure mode the ledger is supposed to eliminate.
- **Recommendation:** Treat canonical ledger registration as a required post-spawn transaction: on failure, mark the lane claim with a loud terminal/recovery-needed row and exit nonzero, or create a durable pending dispatch record before spawn and atomically promote it with the call_id after spawn.

#### 2. [HIGH] Call_id ledger append is O(N) full-file rewrite

- **File:** `src/tac/deploy/modal/call_id_ledger.py` lines 387-428
- **Body:** The ledger helper calls load_call_ids_strict(), appends in memory, then serializes and replaces the entire JSONL file for every event. This is logical append-only, but not append-amortized: at 10000 entries every dispatch/harvest holds the exclusive lock while reading, parsing, serializing, fsyncing, and replacing the full ledger. Under fan-out this can hit the 30s lock timeout, and the dispatcher currently swallows that registration failure.
- **Recommendation:** Use true append under lock with O_APPEND plus strict tail validation, or move the primary index to SQLite/DuckDB with indexed call_id/status queries. Add a 10000-row contention test that proves registration latency and lock wait stay bounded.

#### 3. [HIGH] Lightning lock reentry is process-global, not thread-owned

- **File:** `src/tac/deploy/lightning/active_jobs_state.py` lines 123-131
- **Body:** The active-jobs fcntl wrapper skips reacquiring the file lock whenever _active_jobs_lock_depth > 0. That counter is process-global, not thread-local. If two threads in the same process enter the helper concurrently, the second thread bypasses fcntl entirely and both can load/mutate/save the shared JSON state at once, dropping rows or corrupting job lifecycle state. This is especially risky because the repo explicitly promotes threaded parallel dispatch.
- **Recommendation:** Track lock ownership by thread id with a threading.RLock, or remove same-process bypass except for same-thread reentry. Add a multithreaded stress test that calls register_pending_job_locked/update_pending_to_active_locked concurrently and verifies no rows are lost.

#### 4. [MEDIUM] NVML block gate does not enforce canonical values

- **File:** `src/tac/preflight.py` lines 54040-54053
- **Body:** Catalog #244 only checks whether the three token strings appear anywhere in each script. A comment, a wrong value, a non-export assignment, or a legacy literal that drifts from tac.deploy.modal.runtime would still pass. The hand-written substrate scripts hard-code the current defaults, so runtime.py is not actually a single source of truth once constants change.
- **Recommendation:** Parse the shell export assignments and require exact exports, exact default values from tac.deploy.modal.runtime, and placement before bootstrap/trainer imports. Add a generator golden test plus a legacy-driver drift test that fails when runtime constants change without regenerating/backfilling scripts.

#### 5. [MEDIUM] Vast.ai cleanup treats corrupt tracker as empty

- **File:** `src/tac/vastai_tracker.py` lines 90-113
- **Body:** The read-only Vast.ai loader returns [] for malformed JSON, and the orphan cleanup tool uses that path, prints zero records, and exits successfully. A corrupt tracker can therefore hide active paid instances until some later mutating call happens to quarantine it. That is not fail-closed survival for the corruption scenario.
- **Recommendation:** Expose a strict read path for cleanup and make tools/vastai_orphan_cleanup.py fail nonzero on corrupt tracker state, including pointers to any quarantined tracker files. Only dashboards should use the lossy [] fallback.

**Next steps:**

- Add regression tests for 10000-row Modal ledger contention, threaded Lightning state updates, corrupt Vast.ai cleanup behavior, and exact NVML export drift against runtime constants.
- Decide whether direct experiments/modal_train_lane.py is a supported dispatch entrypoint; if yes, make clean-head or sentinel coverage mandatory by default.

---

### Chunk 10: submissions_inflate_runtime (re-run via `HARVEST-CHUNKED-CODEX-MISSING-AND-AGGREGATE-20260515`)

**Verdict:** `needs-attention`

**Summary:** No-ship: the live working diff only touches the Modal harvest summary, not the named submission runtimes. In that diff, partial zero-artifact recoveries are being recorded as successful completed terminal events, which can hide unrecovered jobs and close the recovery loop with no usable evidence.

**Findings:**

#### 1. [HIGH] Zero-artifact partial harvests are closed as successful completed jobs

- **File:** `experiments/results/_modal_harvest_summary.json` lines 3356-3402
- **Body:** This entry says n_artifacts is 0 and crash_kind is HARVESTED_PARTIAL, but the same record sets outcome to successful_dispatch, appends a terminal claim with status: completed_modal_training_recovered_no_score_claim, and appends terminal evidence. That converts an incomplete recovery with no logs/checkpoint/archive artifacts into a terminal success. The same pattern appears in other added entries, so downstream lane state can be closed and re-harvest suppressed even though there is no artifact custody to inspect or reuse.
- **Recommendation:** Treat HARVESTED_PARTIAL or n_artifacts == 0 as incomplete/failed recovery regardless of provider rc. Do not append terminal evidence or completed terminal claims until a minimum artifact set is present, and use a distinct status such as failed_modal_harvest_partial or recovery_incomplete_needs_retry.

**Next steps:**

- Reclassify all zero-artifact HARVESTED_PARTIAL rows before shipping this ledger.
- Add a guard in the harvester so n_artifacts == 0 cannot produce successful_dispatch or completed_modal_training_recovered_no_score_claim.

---

## Cross-cutting themes (appearing in 3+ chunks)

### Theme A — Declaration-only / token-presence validation (chunks 1, 2, 5, 7, 9)
Multiple gates accept text tokens, metadata booleans, or string paths as proof, instead of cryptographically binding to current archive bytes / runtime hashes / executed verifier output.
- Catalog #220 accepts operational self-attestation via free-text lane evidence (chunk 7).
- Catalog #242 only validates `tac.substrate_registry.example_template`, not the 27 decorated `train_substrate_*.py` files (chunk 1).
- Catalog #244 NVML block check is token-grep, not exact-export-value matching against `tac.deploy.modal.runtime` constants (chunk 5).
- Catalog #272 accepts `byte_mutation_smoke_passes=True` as proof (chunk 1) and accepts stale proof artifacts not bound to current `archive_sha256` (chunk 9).
- `_verify_tier1` token-grep marks autocast/TF32/torch.compile as present from a CLI flag declaration alone (chunk 1).

Cross-ref: CLAUDE.md "Comment-only contracts — FORBIDDEN" + "Apples-to-apples evidence discipline" + HNeRV parity discipline lesson 11 (No-op detector).

### Theme B — Fail-open guards (chunks 2, 4, 6, 8, 10)
Catalog gates that swallow exceptions or emit warnings instead of blocking.
- Catalog #185 catches `TypeError` from gate signature drift and silently skips (chunk 2).
- `_consult_preflight_risk_scorer` builds empty `GateVerdictPanel(verdicts={})` and only logs WARN/REFUSE without aborting (chunks 4 + 6).
- Mandatory pre-deploy harness only warns when `tools/local_pre_deploy_check.py` is missing (chunk 4).
- Modal harvest summary appends `outcome: successful_dispatch` for `n_artifacts == 0` partial recoveries (chunk 10).
- Modal canonical ledger registration wrapped in broad try/except that only prints a warning (chunk 5).

Cross-ref: CLAUDE.md "Forbidden silent-skip cascades (the bootstrap trap)" + Catalog #14 `preflight_loader_format_safety` (fail-loud loader pattern).

### Theme C — Catalog drift between docs and code (chunks 2, 8, 9)
The CLAUDE.md catalog table no longer matches `src/tac/preflight.py` callable surface.
- Catalog #176 misses 63 lambda-wired strict callsites (39 without CLAUDE rows) e.g. `check_no_comment_only_contracts` at line 3606 (chunk 2).
- 6 phantom catalog rows #273-#278 in CLAUDE.md without corresponding `check_*` callable in `src/tac/preflight.py` (chunk 8).
- Catalog #216 referenced as sister gate but no `^216.` numbered row exists (chunk 8).
- Catalog #185 only matches narrow live-count phrase list; rows with non-canonical wording escape (chunk 8).

Cross-ref: Catalog #118 (no duplicate numbers) + #159 (text-matches-strict-value) + #176 (callsite-has-row) + #185 (live-count-zero) — the sister gate quartet that this theme says is itself drifting.

### Theme D — Auto-bypass without independent attestation (chunks 4, 6, 9)
Paired-env discipline (Catalog #199 / #202 / #243) defeated when wrappers auto-set both env vars themselves.
- `tools/run_modal_smoke_before_full.py:748-751` automatically sets BOTH `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1` AND `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1` on dirty trees (chunk 4).
- `_consult_preflight_risk_scorer` REFUSE class never aborts; bypasses the bypass (chunk 6).
- Memory-backed gate (`PACT_MEMORY_DIR`) passes vacuously off original machine (chunk 9).

Cross-ref: CLAUDE.md "Comment-only contracts — FORBIDDEN" pattern (assertion that a contract holds without runtime enforcement IS the bug class).

### Theme E — Concurrency primitives have undeclared invariants (chunks 5, 8)
- Lightning `_active_jobs_lock_depth` counter is process-global, not thread-local; threaded parallel dispatch can bypass the fcntl lock (chunk 5).
- Catalog #157 expected-content-sha256 only checks paths in `--expected`, not every path passed via `--files`; partial coverage (chunk 8).
- `claim_catalog_number.py` truncate-then-write race: process death between truncate and fsync resets the counter (chunk 8).
- `subagent_checkpoint.py read_checkpoints` silently drops malformed JSONL rows (chunk 8).
- Modal call_id ledger O(N) full-file rewrite under lock can exceed 30s timeout at 10K rows under fan-out (chunk 5).

Cross-ref: CLAUDE.md Catalog #131 + #128 + #138 — sister gates that ALL apply to this theme but at different surfaces.

---

## Prioritized fix queue (HIGH severity, deduplicated)

| # | File:line | Finding | Recommendation |
|---|---|---|---|
| H-1 | `src/tac/preflight.py:56580-56589` | Catalog #242 only imports `example_template`; misses 27 decorated trainers | Enumerate all `train_substrate_*.py`; validate each contract via safe isolated import |
| H-2 | `src/tac/preflight.py:58375-58408` | Catalog #272 accepts `byte_mutation_smoke_passes=True` truthy metadata as proof | Parse canonical proof artifact; require per-section mutation + raw-output delta |
| H-3 | `src/tac/preflight.py:45836-45860` | Catalog #176 misses 63 lambda-wired strict callsites | Replace regex with AST-backed preflight graph validator |
| H-4 | `src/tac/preflight.py:408-436` | Clean cache hides warn-only violations; advisory_count ignored on next run | Don't store clean cache when `advisory_count > 0`; treat as cache miss |
| H-5 | `tools/operator_authorize.py:1521-1551` | Risk scorer empty `GateVerdictPanel(verdicts={})`; never aborts on REFUSE | Feed real preflight gate verdicts; raise on REFUSE before claim creation |
| H-6 | `tools/run_modal_smoke_before_full.py:748-751` | Smoke wrapper auto-sets `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1` itself | Don't auto-set attestation; require operator-supplied sentinel hash or nonce |
| H-7 | `tools/cathedral_autopilot_autonomous_loop.py:820-880` | `predicted_dispatch_risk` field exists but `rank_candidates` ignores it | Make risk explicit sort/filter input; refuse candidates above threshold |
| H-8 | `src/tac/autopilot_rudin_daubechies/compressive_landscape.py:197-229` | "L1 recovery" is inverse-distance interpolation, not sparse recovery | Either implement real sparse recovery with diagnostics or downgrade to heuristic |
| H-9 | `tools/verify_distinguishing_feature_byte_mutation.py:429-486` | Verifier instantiates `SectionResult` without required new fields → TypeError | Thread `MutationTarget` through verifier+CLI; populate new fields on every return |
| H-10 | `src/tac/preflight.py:54458-54466` | Catalog #220 accepts free-text "operational" tokens as runtime-effect proof | Require structured proof reference; validate byte-mutation smoke JSON |
| H-11 | `tools/subagent_commit_serializer.py:420-428` | Catalog #157 expected-hash protection only covers paths in `--expected`, not all `--files` | Require `set(expected_content_shas) == set(files)`; enforce one-hash-per-file |
| H-12 | `tools/claim_catalog_number.py:149-157` | Truncate+rewrite race: crash mid-write resets counter; reissues old numbers | Fail closed on empty/corrupt; write via temp + fsync + rename |
| H-13 | `src/tac/preflight.py:46923-46931` | Catalog #185 misses 6 phantom rows #273-#278 (no callable in preflight.py) | Unconditional CLAUDE-row-to-callable check; backfill or remove rows |
| H-14 | `tools/subagent_commit_serializer.py:91-95` | Worktree isolation only defers edit collisions to merge time | Move lock/log to git-common-dir; require merge-helper that rejects overlapping paths |
| H-15 | `src/tac/preflight.py:35316-35340` | Strict memory-backed gate passes vacuously off original machine (CI / fresh checkout) | Make canonical feedback corpus repo-relative; fail closed if missing |
| H-16 | `src/tac/preflight.py:58446-58476` | Catalog #272 accepts stale proof artifacts not bound to current archive sha (sister of H-2) | Require lane to declare candidate archive/runtime identity; reject mismatched proofs |
| H-17 | `experiments/modal_train_lane.py:1393-1419` | Canonical Modal ledger append wrapped in best-effort try/except | Treat ledger registration as required post-spawn transaction; mark recovery-needed on failure |
| H-18 | `src/tac/deploy/modal/call_id_ledger.py:387-428` | Ledger O(N) full-file rewrite per event; 30s lock timeout under fan-out at 10K rows | Use true append (O_APPEND + tail validation) or move to SQLite/DuckDB |
| H-19 | `src/tac/deploy/lightning/active_jobs_state.py:123-131` | `_active_jobs_lock_depth` process-global, not thread-local; bypasses fcntl in threaded dispatch | Track lock ownership by thread id; threading.RLock or remove same-process bypass |
| H-20 | `experiments/results/_modal_harvest_summary.json:3356-3402` | Zero-artifact `HARVESTED_PARTIAL` rows recorded as `outcome: successful_dispatch` | Treat `n_artifacts == 0` as incomplete regardless of provider rc; use `recovery_incomplete_needs_retry` |
| H-21 | (chunk 6 cross-ref to H-5) Preflight REFUSE cannot block dispatch — same root cause as H-5 | (deduplicated) | (deduplicated) |

---

## Op-routables (ranked by EV)

### OP-1 (highest EV) — Promote Catalog #176 to AST-backed graph validator + backfill 6 phantom rows #273-#278
**Surfaces:** chunks 2, 8.
**Why high EV:** Catalog #176 + #185 + #159 + #118 are the META-meta gates that make the CLAUDE.md catalog table the canonical strictness ledger. Empirical: 39 of 63 strict callsites missing CLAUDE rows; 6 phantom rows without callables. Cost: ~3-5 hr editor, $0 GPU.

### OP-2 — Fix Catalog #272 byte-mutation verifier (`tools/verify_distinguishing_feature_byte_mutation.py:429`)
**Surfaces:** chunks 1, 7, 9.
**Why high EV:** This is the EVIDENCE GENERATOR other gates need. Without it, Catalog #220, #272, and the operational-mechanism declaration are all self-attestation surfaces. Cost: ~2-4 hr editor, $0 GPU.

### OP-3 — Wire `predicted_dispatch_risk` into `rank_candidates`
**Surfaces:** chunks 4, 6.
**Why high EV:** The Rudin-Daubechies autopilot ranker landed last week with full continual-learning loop, but the risk field is dead in the ranker. Cost: ~1-2 hr editor, $0 GPU.

### OP-4 — Fix paired-env auto-bypass in `tools/run_modal_smoke_before_full.py:748-751`
**Surfaces:** chunk 4.
**Why high EV:** Catalog #199 + #202 paired-env discipline EXISTS specifically to prevent silent bypass, but the smoke wrapper auto-sets BOTH env vars. Cost: ~1 hr editor, $0 GPU.

### OP-5 — Promote Catalog #185 to fail-closed on TypeError + canonical wording-agnostic live-count parsing
**Surfaces:** chunks 2, 8.
**Why high EV:** META-meta-meta gate that prevents Catalog drift IS the gate that's most subject to drift. Cost: ~2 hr editor, $0 GPU.

### OP-6 — Atomicize `claim_catalog_number.py` truncate+rewrite to temp+fsync+rename
**Surfaces:** chunk 8.
**Why moderate-high EV:** Crash mid-write reissues old catalog numbers. Already empirically observed: catalog #183/#184 dual-claim collision 2026-05-09. Cost: ~30 min editor, $0 GPU.

### OP-7 — Fix `_active_jobs_lock_depth` thread-safety in Lightning state
**Surfaces:** chunk 5.
**Why moderate EV:** Process-global counter bypasses fcntl in threaded dispatch. Cost: ~1 hr editor, $0 GPU.

### OP-8 — Fix Modal harvest summary `n_artifacts==0` → `successful_dispatch` mislabeling
**Surfaces:** chunk 10.
**Why moderate EV:** Closes recovery loop with no usable evidence; suppresses re-harvest. Cost: ~30 min editor, $0 GPU.

### OP-9 — Promote `_consult_preflight_risk_scorer` to populate `GateVerdictPanel` from real staged-file results
**Surfaces:** chunks 4, 6.
**Why moderate EV:** Sister of OP-3. The risk scorer Catalog #273-#278 are referenced as ship-blockers but have no real consumers (chunk 8 phantom rows finding). Cost: ~2-3 hr editor, $0 GPU.

### OP-10 — Land canonical Modal ledger append-amortization (SQLite or O_APPEND + tail validation)
**Surfaces:** chunk 5.
**Why moderate EV:** Pre-emptive harden BEFORE the bug bites at 10K events. Cost: ~4-6 hr editor, $0 GPU.

---

## 6-hook wire-in declaration per Catalog #125

This aggregation memo is text-only adversarial review consumption. Hook applicability:

1. **Sensitivity-map contribution** — N/A — text-only ledger; no per-tensor importance signal produced.
2. **Pareto constraint** — ACTIVE — every HIGH/MEDIUM finding is a PRE-DISPATCH CONSTRAINT on the operator's next paid GPU dispatch.
3. **Bit-allocator hook** — N/A — no per-tensor allocation change proposed.
4. **Cathedral autopilot dispatch hook** — ACTIVE — OP-3 wires `predicted_dispatch_risk` into `rank_candidates`; OP-9 populates `GateVerdictPanel` for the autopilot's risk scorer.
5. **Continual-learning posterior update** — N/A with rationale: this aggregation does not produce empirical anchors; sister continual-learning posterior loop runs from the codex companion's own session-level audit ledger.
6. **Probe-disambiguator** — N/A with rationale: codex's adversarial-review verdict format is single-verdict-per-finding; no two defensible interpretations were surfaced where ship/no-ship probe disambiguation would apply.

---

## Sister-subagent ownership-honored note per Catalog #230

Scope: READ-ONLY codex invocations + READ existing chunk logs + WRITE-NEW to `.omx/research/chunked_codex_review_chunk{5,10}_20260515.log` + this aggregation memo + sister landing memo + lane registry mark via canonical helper. NO source files / preflight.py / contended files were modified.

In-flight sister subagents:
- `a21e31a4` (FIX-CODEX-WAVE-3 — touches `preflight.py` + 2 dispatch tools) — UNTOUCHED.
- `aba0abf5` (OMNIBUS-BUG-CLASS-AUDIT — read-only audit) — UNTOUCHED (read-only scopes are non-overlapping).

---

## Premise verification per Catalog #229

- PV-1: 8 chunk log files exist on disk via `ls .omx/research/chunked_codex_review_chunk*_20260515.log` (expected chunks 1, 2, 3, 4, 6, 7, 8, 9). CONFIRMED.
- PV-2: Codex companion exists at `/Users/adpena/.claude/plugins/cache/openai-codex/codex/1.0.3/scripts/codex-companion.mjs`. CONFIRMED.
- PV-3: No predecessor checkpoint for this `subagent_id` via `tools/subagent_checkpoint.py read`. CONFIRMED (fresh start).
- PV-4: Chunk 3 final assistant message is identical to chunk 1 (codex companion ran adversarial-review against working tree diff WITHOUT per-chunk path scoping). CONFIRMED via line-by-line comparison of final rendered sections.
- PV-5: Chunks 5 + 10 codex calls completed cleanly with `# Codex Adversarial Review` rendered sections + `Verdict: needs-attention` + structured findings. CONFIRMED via `grep -n "^# Codex Adversarial Review"`.
- PV-6: Existing partial ledger at `.omx/research/codex_chunked_full_codebase_review_20260515.md` from orchestrator `afe253c` (313 lines, chunks 1/2/4/6/7/8/9 verbatim). CONFIRMED.

---

## Reactivation criteria

Per CLAUDE.md "Forbidden premature KILL": every finding above is DEFERRED-pending-fix, NOT KILLED. Reactivation criteria for re-validating each finding:

1. **For each HIGH finding** — re-run the cited code path against synthetic inputs designed to trigger the bug class; if the bug is no longer reproducible AND a sister STRICT preflight gate exists at 0 live violations, the finding may be marked CLOSED in a follow-up ledger.
2. **For each MEDIUM finding** — same as HIGH but the strict-preflight requirement may be waived if the cost of a strict gate exceeds the empirical cost of the bug class.
3. **For dropped chunk 3 duplicate** — if codex companion gains per-chunk `--paths` scoping, re-run chunks 3 + 1 with explicit path filters and compare verdicts. Currently chunk 3 carries no unique evidence beyond chunk 1.
