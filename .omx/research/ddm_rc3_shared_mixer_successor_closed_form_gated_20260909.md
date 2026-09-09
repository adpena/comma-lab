# ddm_rc3 — closed-form gated shared model-row mixer

Date: 2026-09-09. Tokens: `[no-triality] [p0-ledger-ok]`.
Axis: `[macOS-CPU advisory / scorer-free EXACT byte measurement]`.
`research_only=true`, `score_claim=false` until MAIN's exact T4 replay.

**SEAL READY: /Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor/SEAL_ddm_rc3_shared_mixer_successor_contest_cuda.json**

The retained candidate is **181,213 B**, **201 B smaller** than the live 181,414 B rc2 archive.
Its SHA-256 is `bea79912d3b2230ebfb2434190f9fdbef18d16d8fa46f5e647c82a21ee11a6b9`. Exact decoded IHS1 identity plus unchanged semantic,
carrier and token-tail sections prove the lossless composition. The projected score is
**0.13871672690067088**, using unchanged distortion; it is not a new contest measurement.
The exact frontier is UNMOVED by this arm. MAIN owns the authorized T4 fire; no scorer or Modal ran here.

## RECALL EVIDENCE

Read the charter and common contract in full, PROGRAM.md, governing CLAUDE.md/AGENTS.md,
operating manual, live hot state, canonical pointer and lane registry. The initial
`tools/subagent_checkpoint.py read --subagent-id ddm_rc3` found no predecessor record.
CLAUDE.md and AGENTS.md were byte-identical at intake. The common contract's August frontier
paragraph is stale; the canonical pointer's move 33 is the live authority.

Content recall searched `.omx/research/` memos and arm receipts using
`IHS1|shared.{0,20}mix|model_section_adaptive_recode_ceiling|order.?2.{0,30}model`,
then `entropy|recode|IHS1|model|bound`. It also searched CANONICAL_RESEARCH_INDEX*,
sub015_DAG_* FEED blocks, design/SPEC documents and canonical_task_status.jsonl using
`IHS1|ddm_rc[123]|shared logistic`. The equations registry was enumerated with
`.venv/bin/python tools/list_canonical_equations.py --json`, selecting
`model_section_adaptive_recode_ceiling_v1`, `coder_strength_substitutes_for_capacity_v1`
and `hpac_prior_capacity_slope_v1`.

Beyond the charter seeds:

- `ddm_mz1_model_section_rate_race_20260815.md` reinforced retaining actual framed bytes;
  its older-object negative was not transferred to this model.
- `ddm_dcf1_duplicate_carry_factorization_20260831.md` distinguished model content from
  framing redundancy; this arm keeps the outer header grammar and all other sections intact.
- `experiments/ddm_fx1_logistic_mixer_corrector.py` supplied the causal shared-update
  precedent. RC3 instead reuses RC2's already-shipped integer stretch/squash decision surface.
- The task ledger's rc1 follow-ons distinguish model-prior work from the separate token-tail
  consumer. No tail recode or capacity work was duplicated.
- `ddm_scg2_seal_custody_followon_digest_naming_20260909.md` resolved the two runtime-hash
  definitions: the seal's `tac.candidate_seal.measure_runtime_digest` is named explicitly,
  separate from the upload-manifest digest in the frontier mirror.

The index/DAG/design searches did not find another current-object position/sibling/order-2
IHS1 shared mixer beyond the charter seeds in those searched surfaces. The current registry
contained rc2's shared-mixer anchor, which this result extends.

## Verified source and denominators

The live tree was read-only:
`/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/candidate_runtime`.
Archive SHA `c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e`, 181,414 B.
The source receiver restored **17,770 B IHS1**, SHA
`817281908d993d89f62349fa466ad53f204b7e83d36e29ba0edc30cf2cb8f085`.
The complete object contains **517 rows, 20,416 signed values, 89,241 coded bits**;
3,355 zero-depth values cost no coded bits, leaving 17,061 nonzero-depth symbols.

Source-number recheck, from retained bytes rather than prior scalar summaries:

| Quantity | Bytes |
|---|---:|
| Live RC2 HPAC container | 12,112 |
| Live RC2 range payload / weights / J | 9,419 / 8 / 9,427 |
| RC2-convention order-1 Miller–Madow estimate | 8,469.343439 |
| Live 8-weight J minus that estimate | **957.656561** |
| Prior unshipped 16-weight range / weights / J | 9,406 / 16 / 9,422 |
| Prior 16-weight J minus the estimate | **952.656561** |
| Retained predecessor HPAC minus live HPAC | 12,343 − 12,112 = **231** |

Thus the charter's “953 B remaining” used the unshipped 16-weight raw-J winner.
The live 8-weight denominator leaves 957.657 B. `SOURCE_NUMBER_RECHECK.json` preserves this correction.

## Closed-form leg, before any new encoder

`ENTROPY_PRICING.json` and `CLOSED_FORM_GATE.json` were written before the first RC3 encode;
the gate hash is `b328c7132e5e001b355a562b8e053fa13685ca3aac19352d67472d9f56df5761`.
For a fixed context C, the saturated empirical static-model log-loss minimum is
`L_plugin = sum_c [n_c log2(n_c) − sum_x n_cx log2(n_cx)]`.
The asymptotic correction is `L_MM = L_plugin + sum_c(K_c−1)/(2 ln 2)`.
The first symbol at each depth is marginal-priced for the inherited 8,469.343439 B convention.
A saturated empirical minimum is an oracle diagnostic; **Miller–Madow is an estimate,
not a rigorous finite-sample entropy bound or an attainable archive size**.

Position means row-position quartile. Sibling group means a maximal consecutive run of
rows with equal known width; sibling value means the preceding such row's decoded value
at the same column. These are causal geometry, not hidden learned tables.

| Context | Empirical oracle log-loss B | Full MM diagnostic B | Symbols passing support screen |
|---|---:|---:|---:|
| depth | 9607.78 | 9640.33 | 15,791/20,416 |
| order2 | 4343.49 | 4987.02 | 4,779/20,416 |
| order2_coarse | 7426.91 | 7962.06 | 6,453/20,416 |
| position | 9315.76 | 9408.64 | 12,792/20,416 |
| prev | 8084.84 | 8465.17 | 8,280/20,416 |
| prev_group | 7094.60 | 7675.28 | 5,399/20,416 |
| prev_position | 6905.25 | 7528.41 | 5,382/20,416 |
| prev_position_group | 5338.12 | 6088.06 | 4,022/20,416 |
| sibling_group | 9442.58 | 9527.43 | 11,600/20,416 |

The screen requires context n ≥ 5 occupied outcomes and every observed outcome count ≥2.
It is a declared asymptotic-screening heuristic, not a confidence guarantee. Sparse full
order-2 estimates above are **not used as supported bounds**. Restricting both the fine
context and previous-only comparison to the SAME supported, nonzero-depth symbols gives:

| Context refinement | Supported coded symbols | Matched-subset plug-in gain B | MM gain estimate B |
|---|---:|---:|---:|
| order2 | 1,426/17,061 | 7.138 | 4.163 |
| order2_coarse | 3,100/17,061 | 10.685 | 4.734 |
| prev_group | 2,045/17,061 | 13.368 | 10.843 |
| prev_position | 2,028/17,061 | 8.446 | 2.946 |
| prev_position_group | 680/17,061 | 9.078 | 6.373 |

The prior prediction that ≥600 B is demonstrably order-2-reachable is **not established**.
The unsupported cells are not extrapolated. This does not close the shared-mixer family.

The separate counted-design forecast starts at the actual RC2 weights, uses causal expert
log odds X, computes g = Xᵀ(y−p), H = Xᵀdiag(p(1−p))X, and solves one closed-form local
quadratic step `delta = (H+I)^−1 g`. Forecast gain is
`(gᵀdelta − deltaᵀHdelta/2)/(8 ln 2)`. This is explicitly a local quadratic forecast,
**not an entropy bound**. A second, quantized log-loss replay sanity-checks that forecast.
The outer-container offset is held at RC2's measured value solely for prediction.

| Family | Predicted range B | int8 / fp16 parameter B | Context-table B | Predicted HPAC B | Predicted net vs 12,112 B | Quantized replay net B | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| position_sibling | 9234.43 | 25 / 49 | 0 | 11946.43 | 165.57 | 190.94 | ADMIT |
| order2 | 9233.65 | 25 / 49 | 0 | 11945.65 | 166.35 | 188.00 | ADMIT |

Each design has 24 globally shared initial weights and one counted learning-rate byte;
the header costs 2 B more than RC2. The implemented scope is int8, two families. Both
forecasts clear 150 B; only these two families were built. No credit for future online
learning was included in admission. An independent scalar-order einsum recomputation
verified the initially warning-emitting Accelerate matrix operations: gradient differences
≤3.32e−11 and Hessian differences ≤5.46e−12, all finite (`NUMERICAL_RECHECK.json`).

## Actual race and container objective

Each family was fit separately with bounded L-BFGS-B and two integer coordinate passes.
All intermediate fits, weight vectors and real 89,241-event matrices were retained.
The coder has RC2's eight predictors plus four adaptation timescales, KT count predictors,
causal geometric/history contexts and one shared intercept. The decoder's mixture and
online weight updates use integer arithmetic only. No scorer weights or video-derived
constants enter receiver code.

Five rate settings per family were tested: frozen weights (0), and counted dyadic
update shifts 18/20/22/24. This is a bounded tuning domain, not a proof of global optimality.
For every row: pack → encode twice → exact rider identity → fresh decode to the full
IHS1 body. Static-case encoder event matrices match the independent forecast instrument exactly.

| Trial | Range B | J B incl. 25 parameters | q11/w24 HPAC B | Best shippable container B | Best shape | Net B |
|---|---:|---:|---:|---:|---|---:|
| position_sibling_lr0 | 9,198 | 9,223 | 11,933 | 11,911 | q10/w22 | +201 |
| position_sibling_lr18 | 9,179 | 9,204 | 11,949 | 11,925 | q10/w22 | +187 |
| position_sibling_lr20 | 9,190 | 9,215 | 11,965 | 11,915 | q10/w22 | +197 |
| position_sibling_lr22 | 9,197 | 9,222 | 11,943 | 11,929 | q10/w22 | +183 |
| position_sibling_lr24 | 9,244 | 9,269 | 11,980 | 11,973 | q10/w22 | +139 |
| order2_lr0 | 9,202 | 9,227 | 11,940 | 11,940 | q11/w22 | +172 |
| order2_lr18 | 9,178 | 9,203 | 11,916 | 11,916 | q11/w22 | +196 |
| order2_lr20 | 9,191 | 9,216 | 11,945 | 11,925 | q10/w22 | +187 |
| order2_lr22 | 9,199 | 9,224 | 11,949 | 11,949 | q11/w22 | +163 |
| order2_lr24 | 9,244 | 9,269 | 11,975 | 11,973 | q10/w22 | +139 |

The raw-J winner is **order2_lr18, 9,203 B**. The archive winner is instead
**position_sibling_lr0, 11,911 B HPAC** at ck2=true/q10/lgwin22. The selected design's
predictors remain online-adaptive, while its shared mixture weights are fixed at the
counted fitted values. The online weight learner was implemented and tested; it lost
the final archive selection.

All **198 container cells** are retained: 11 objects (RC2 control plus ten new trials)
× two ck2 settings × q9/q10/q11 × windows 22/23/24. All decompress exactly.
The source control reproduces the live stream byte-for-byte only at its actual q11/w22
shape; equal-sized other windows have different header bytes. Ties prefer the
byte-identity-proven shipped shape. ck2=false stays measurement-only because the live
RX1 flags cannot represent it unambiguously. It is never staged.

The selected 24-weight position/sibling row was predicted to save 165.567 B and measured
**201 B at the archive** (forecast residual +35.433 B). Relative to its q11 container,
retuning the edited body adds 22 B. Retuning the unchanged RC2 body adds 0 B.
The prior 200–400 B net band is met at 201 B; the separate ≥600 B support claim remains unproved.

## Staged receiver, public proof, and seal

The live pointer was re-read before staging and sealing. Only these owned-tree files differ:
`archive.zip`, `MANIFEST.sha256`, `inflate.py` archive pins, `runtime/ihs2.py`,
`runtime/residual_archive.py`, and new `runtime/rc3_shared_mixer.py`.
Only the HPAC section and its length field change in the archive.

The staged codec is byte-identical to the reviewed encoder source. After staging,
two fresh encodes and complete archive packs again produced identical archives.
The staged public `read_residual_archive` → `materialize_ihs1` path wrote its decoded
17,770-byte body to retained storage and it matched the source exactly.

| Unchanged section | Bytes | SHA-256 |
|---|---:|---|
| semantic | 30,246 | f1f0f85730981f2639edea2fdb36bf190c6c065a0c91b15afe4f6ffd0f99b186 |
| carrier | 18,621 | fa18c86fe9158a4bb9a22df47f1ee2a68948c25c8cbbb865877cd40ac438064d |
| token tail | 120,321 | af6b0997bf26c446f2b235d8cad7a83b927c17a0bb53b40e18ef4e05ff4832dc |

Both candidate and live-control `runtime.f26_inflate.inflate_archive` runs recorded an
actual Python call event in the intended `runtime/residual_archive.py::decode_production_tokens`,
then ran without an exception until the 60-second process-group bound. Each attempt has
a unique retained marker and stdout/stderr. Timeout alone never counts as path evidence.
Both actual `bash inflate.sh` runs reached the expected linux-nvidia-t4 CUDA gate after
accepting archive pins. These are bounded local entrypoint proofs, not a complete local
video decode or a T4 runtime measurement. No local scorer was invoked.

`PUBLIC_ENTRYPOINT_SMOKE.json` passes `tac.candidate_seal._public_smoke_problems` with
zero problems; `tools/make_candidate_seal.py` produced **SEAL_VALID** for contest_cuda.
Seal SHA `b4026560bb8b67df1b206bfc810a14e09d197277fd9f6eb93dfb4a36c49c5a93`. Runtime digest
`c38f79572f585b66cf09c2bd365310e4c150a2db1c718666ab77e4d42bf5d13a` is explicitly `tac.candidate_seal.measure_runtime_digest`.
Seven receiver files are pinned. The admission bar is net dS < −0.0001 with zero
pointer tolerance, and the base report-8dp bound is derived from its actual T4 receipt.

Rate arithmetic only: `delta S = −201 × 25/37,545,489 = −0.00013383764957755645`.
The composed prediction is 0.13871672690067088. Neither value is promoted to a measured score.

## Retention, validation, and landing custody

All bulk is under `/Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor`. Storage preflight checked the first-choice SSD (about 16 GiB
available at intake); no fallback routing or local bulk was needed. Source payloads,
all ten riders/twins/parameter sets/decoded bodies, all 198 Brotli cells, staged archive
twins, parsed public body, smoke artifacts, code snapshots and malformed-input controls
remain in custody. `RESULT.json` indexes the evidence and `RETENTION_INVENTORY.json`
hashes the files. No materialized candidate payload was discarded.

Resume fixes bind fit inputs, gate, source, dependency code, trial parameters, container
inputs and runtime digests. A receipt interrupted before its binding is preserved with
custody metadata and replayed; partial runtimes are preserved before rebuilding the stage.
Compiler, decoder and smoke child groups are killed and reaped on timeout. No detached
process, clock waiter, scorer, upstream write or live/sibling-tree write was used.

Validation: ten real-data twin encodes and exact decodes; two post-stage twin encodes;
full unchanged-section census; two public smoke pairs; five real-rider malformed-header/
length controls refused; Ruff passes. Independent adversarial review found evidence-resume
and stale-smoke-marker defects, which were fixed before landing. A final independent
static pass is CLEAN; the author separately reviewed the final arithmetic and retention paths.
The codec bytes did not change after their real-data proof.

Commit is through the serializer with post-edit hashes and no attribution trailer.
If sandbox Git writes refuse, the fallback bundle/patch/receipt is the handoff to MAIN;
that is **not a landed main-branch commit**. Shared equation/lane writes are made through
the canonical helpers; the arm-owned evidence copy is included without absorbing other
arms' dirty files.

## Canonical equations leg and integration

`model_section_adaptive_recode_ceiling_v1` receives anchor
`rc3_closed_form_shared_mixer_and_container_objective_20260909` through
`update_equation_with_empirical_anchor`. It records the source-denominator correction,
sparse-estimator limitation, forecast vs measured 201 B, and raw-J/container ranking reversal.
The arm-owned anchor receipt retains the event for MAIN's landing consumer.

This is research-only until exact replay. Sensitivity and Pareto contributions are the
measured zero-distortion rate delta; per-tensor bit allocation is unchanged; the dispatch
consumer is the validated candidate seal owned by MAIN; posterior calibration consumes
the equation anchor. No new autonomous dispatcher or unrelated control policy is created.
# NO_SUPERSESSION_NEEDED: rc2 remains the measured frontier; this is its unmeasured sealed successor.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store `/Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor/SEAL_ddm_rc3_shared_mixer_successor_contest_cuda.json`.** Fire one uniquely claimed contest-CUDA T4 n600 lane when the live pointer still names `c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e` and this seal validates. If another arm lands first, rebase only this HPAC section and reader on its tree, repeat the public proof and reseal before firing. Admit only if exact distortion is unchanged and net dS < −0.0001.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store the serializer fallback bundle and receipt under `/Volumes/VertigoDataTier/pact/ddm_rc3_shared_mixer_successor/serializer_fallback`.** If the serializer returns a sandbox Git refusal, land the exact reviewed exclusive-file snapshot from that receipt in a Git-writable session; replay the retained equation event through the canonical helper without absorbing sibling work.

## LIVE-HYPOTHESES

- More low-parameter shared predictors may convert remaining model-row redundancy: a 24-weight
  model now converts another 201 archive bytes. A successor requires the sealed T4 result and
  a fresh counted closed-form forecast of ≥150 B before implementation; this is a dormant
  research hypothesis, not an active launch order.
- Fine online-learning rates or another outer container may change selection: the online
  order-2 row wins raw J, while q10 makes the fixed-weight geometry row win the archive.
  This is untested outside the explicit tuning domain and carries no assumed byte credit.

## DEAD-ENDS

- **FOLDED, FORMULATION:** treating the sparse full order-2 MM value as a supported ≥600 B
  opportunity; most contexts fail the support screen, and the matched supported subset
  establishes only 4.163 B of corrected incremental structure on 1,426/17,061 coded symbols.
- **FOLDED, INSTANCE:** selecting by raw J or keeping q11 by default; both select the wrong
  candidate in this race. The measured archive objective chooses q10 geometry mixing.
- **FOLDED, FORMULATION:** shipping ck2=false through the existing RX1 flags; it is unrepresentable
  without a separate grammar change. Semi-static tables and capacity sweeps remain inherited
  scoped negatives and were not rerun.

OWN-VEHICLE FRONTIER: S 0.13885056455024844 @ 181,414 B [contest-CUDA T4 n600], unchanged by this arm.
