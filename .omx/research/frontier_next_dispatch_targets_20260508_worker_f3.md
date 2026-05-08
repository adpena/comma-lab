# Frontier Next Dispatch Targets - 2026-05-08 worker_f3

Scope: identify the next exact-evaluable score-lowering targets after the
current tranche, without code changes. This is a dispatch-routing ledger, not a
score claim.

## Evidence Rules Applied

- Current score truth is exact CUDA auth eval through
  `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- Active local A++ floor used here:
  `score=0.20898105277982337`, `archive_bytes=185578`,
  `archive_sha256=ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`,
  from `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`.
- Active floor component budget:
  - seg term: `0.067082`
  - pose term: `0.01833030277982336`
  - rate term: `0.12356875`
  - non-rate term: `0.08541230277982336`
- Rate-only candidates must beat `185578` bytes on the active scorer-equivalent
  packet before exact-CUDA spend is justified. PR101-byte candidates below this
  byte count are not automatically score-lowering because PR101 exact semantics
  have worse pose contribution than the PR103-on-PR106 floor.
- Scorer-changing candidates require byte closure, runtime-consumption proof,
  strict preflight/compliance, Level-2 dispatch claim, exact CUDA artifact, and
  adversarial result review before status changes.
- Active same-lane dispatches must not be duplicated. Claim hygiene is part of
  the evidence standard.

## Current Anchors

### PR103-on-PR106 A++ floor

- Exact artifact:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- Archive:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip`
- Bytes/SHA:
  `185578` /
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- Runtime tree SHA:
  `54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6`
- Interpretation: current rate anchor and score floor. All rate-only work is
  subordinate to this floor.

### PR101 / PR106 monolithic HNeRV

- PR101 exact replay remains a useful semantic control, but not the current
  score floor:
  `score=0.22635331443973267`, `bytes=178258`,
  `archive_sha256=b83bf348ab7d0c8000b79b8e24f24d43cd6c88c53c39ec23b9da411c7fc7d557`.
- PR106/PR106x remains a predecessor/control packet for monolithic section
  surgery. It is not the floor after PR103-on-PR106.
- Monolithic anatomy is one-member ZIP payload surgery, not member-level
  component surgery. Valid byte claims must name internal parser-proven
  sections, offsets, lengths, SHA-256s, and runtime consumption.

### Lossy coarsening exact negative

- Exact artifact:
  `experiments/results/pr101_lossy_decoder_coarsening_direct_exact_20260508/auth_eval_work/contest_auth_eval.json`
- Archive:
  `experiments/results/pr101_lossy_decoder_coarsening_direct_exact_20260508/static_runtime_release/archive.zip`
- Bytes/SHA:
  `156404` /
  `ab8a8a13c70ba8107f6260c973ce616d8fca3447b58ec669dd524658aa1c4671`
- Exact score:
  `0.351718793322788`
- Classification: A-negative for the measured config
  `per_tensor_K_budget=0.05`, not a family kill.
- Consequence: rel_err-only proxy is not a dispatch criterion. Any lossy
  decoder-coarsening stack must add score-aware allocation, lower-risk budgets,
  QAT/fine-tune evidence, tensor whitelist/blacklist evidence, or a separate
  contest-CUDA anchor before another exact spend.

### Active arch_shrink Lightning

- Active claim/job:
  `arch_shrink_x0.4_lightning` /
  `arch-shrink-x0-4-lightning-20260508T024304Z`
- Status in inspected ledgers/state: running Stage 2 training, no terminal
  `archive.zip`, no terminal `contest_auth_eval.json`, no terminal claim row.
- Expected local artifact root:
  `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z`
- Interpretation: already dispatched. Do not duplicate.

## Ranked Top 5 Next Actions

### 1. Harvest and adjudicate active `arch_shrink_x0.4_lightning`

Type: scorer-changing architecture/retrain. Not rate-only.

Why it ranks first:

- It is the only inspected lane already under a live Level-2 dispatch claim and
  running in the exact-eval pipeline.
- The cathedral catalog carries an empirical byte anchor near `83571` bytes.
  That byte count would have a rate term near `0.05565`, leaving roughly
  `0.15333` non-rate budget to beat the active `0.20898105277982337` floor if
  the trained packet avoids component collapse.
- No wall-clock is saved by relaunching; the shortest path is harvest, not
  duplicate dispatch.

Exact blockers:

- Terminal `archive.zip` absent.
- Terminal `contest_auth_eval.json` absent.
- Terminal exact-CUDA log absent.
- Claim remains active, not completed.
- Current state is `in_progress`, not score evidence.

Required proof/artifacts before any score movement:

- Final archive path, byte count, archive SHA-256, member SHA-256s.
- Final `contest_auth_eval.json` or adjudicated JSON with `n_samples=600`,
  CUDA/T4-equivalent hardware, component fields, and recomputed formula score.
- Runtime tree SHA and scored `inflate.sh` closure.
- Terminal dispatch-claim row for
  `arch-shrink-x0-4-lightning-20260508T024304Z`.
- Adversarial result review classifying the result as legitimate movement,
  component collapse, runtime bug, infrastructure failure, or indeterminate.

Shortest wall-clock path:

1. Poll the active job only; do not relaunch.
2. When terminal artifacts appear, rsync/harvest into the existing expected
   artifact root.
3. Recompute formula from component fields and compare against the active
   floor.
4. Append terminal claim row and result-review ledger.

Dispatch decision:

- Do not start another dispatch. This lane is already active.

### 2. Build the corrected cross-paradigm ADMM-K plus Op1 packet into a real runtime archive

Type: scorer-changing. It changes decoded payload behavior and is not a
rate-only repack.

Why it ranks second:

- The corrected substrate row reports `137469` bytes for
  `cross_paradigm_admm_continuous_k_then_op1_finalizer_substrate_corrected`,
  which would have a rate term near `0.09154`.
- At that rate, the packet can beat the active floor if non-rate terms stay
  below roughly `0.11744`.
- The current evidence is closer to a real path than planning-only omega
  stacks because it has a corrected materialized substrate and dequant formula
  review, but it is still not a contest archive.

Exact blockers:

- `137469` is still byte-proxy/substrate evidence, not a scored archive.
- No real `archive.zip` carrying the corrected stream and all charged side
  information.
- No `inflate.sh`/runtime consumer proven to decode and consume the corrected
  ADMM-K plus Op1 stream.
- No old/new parser-proven section replacement manifest on a scorer packet.
- No runtime-consumption proof, strict compliance proof, Level-2 claim, or
  exact CUDA artifact.
- The lossy coarsening exact negative invalidates rel_err-only confidence.

Required proof/artifacts:

- Byte-closed contest archive with path, bytes, SHA-256, member names, member
  SHA-256s, and internal stream offsets/lengths/SHAs.
- Runtime that consumes the new stream through the scored inflate path.
- Runtime-consumption proof binding archive SHA, changed sections, generated
  outputs, command log SHA, and runtime tree SHA.
- Score-affecting payload change manifest with old/new archive and section
  SHAs.
- Strict pre-submission compliance pass.
- Level-2 claim before exact eval.
- Exact CUDA auth eval and adversarial review.

Shortest wall-clock path:

1. Use the monolithic packet bridge discipline: replace parser-proven sections,
   not imagined member-level components.
2. Materialize the corrected ADMM-K plus Op1 stream into a PR101/PR103/PR106
   runtime packet with all side information charged.
3. Prove local decode and runtime consumption without scorer loads at inflate.
4. Run the closure/floor gate. Dispatch only after the packet is byte-closed
   and runtime-consumed.

Dispatch decision:

- Do not dispatch the existing `137469` row. It lacks runtime proof and is not
  a real scored archive.

### 3. Requalify `admm_x_lossy_coarsening_path_b_step6_no_dead_k` before exact CUDA

Type: scorer-changing lossy packet. Not rate-only.

Why it ranks third:

- This is closer to exact-evaluable than the corrected `137469` substrate
  because it has a byte-closed archive:
  `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/submission/archive.zip`
- Archive bytes/SHA:
  `153671` /
  `b7b09089e852872bd67b4b8aa04c1b4d46168bb89343acff81796c5551d63d05`
- Its rate term is near `0.10233`, leaving roughly `0.10665` non-rate budget
  to beat the active floor.
- It removes dead K bytes and records `score_affecting_payload_changed=true`.

Exact blockers:

- Current manifest marks `ready_for_exact_eval_dispatch=false`.
- Current evidence is CPU/proxy build evidence, not score evidence.
- It explicitly requires contest auth eval and an `apogee_int6` contest-CUDA
  anchor before promotion.
- Its `rel_err_actual_int8=0.04153796782863332` is worse than the exact-negative
  lossy coarsening run's `0.03855950900557584`, which already scored badly.
- No Level-2 claim for a new exact eval.
- Need strict compliance and runtime closure rechecked against the exact scored
  release surface before dispatch.

Required proof/artifacts:

- Confirmed archive bytes/SHA and submission runtime tree SHA for the scored
  release surface.
- Strict pre-submission compliance pass on the exact packet.
- Score-aware reactivation evidence: lower-risk budget ladder, tensor
  whitelist/blacklist, QAT/fine-tune evidence, or CUDA component sensitivity
  support.
- Explicit comparison to the A-negative lossy result and why this packet should
  not repeat the same component collapse.
- Level-2 dispatch claim.
- Exact CUDA auth eval plus result-review packet.

Shortest wall-clock path:

1. Treat the existing archive as a candidate body, not a dispatchable score
   candidate.
2. Add the missing score-aware reactivation gate using the lossy negative as
   the calibration failure.
3. If the gate passes, run strict compliance, claim the lane, and dispatch one
   controlled exact CUDA eval.

Dispatch decision:

- Do not dispatch immediately. It has a real archive, but the exact-negative
  blocker is stronger than the byte gain until reactivation evidence exists.

### 4. Close `categorical_hpm1_refresh` parity and runtime consumption

Type: scorer-changing categorical/runtime packet. Not rate-only.

Why it ranks fourth:

- Hidden-gem routing records a byte-closed candidate at `179979` bytes with
  archive SHA
  `9bfea530f028b342f21ae60d1501f0fc9fc8a4f21f7637c3bba0473ff6303ea3`.
- The byte count is below the active byte floor, but the candidate is
  scorer-changing, so byte count alone is insufficient.
- If categorical/HPM1 semantics preserve or improve SegNet/PoseNet terms, it
  can be exact-evaluable and score-lowering.

Exact blockers:

- HPM1 decode/reencode parity has not passed.
- Known entropy mismatch starts at frame/group/symbol evidence such as
  `frame 0 group 10 symbol 191`.
- Full decode/reencode byte parity is missing.
- Semantic runtime-output parity is missing.
- Runtime-consumption proof is missing.
- No strict compliance pass, Level-2 claim, or exact CUDA artifact.

Required proof/artifacts:

- Full HPM1 decode/reencode parity manifest over the scored stream.
- Runtime-output semantic parity or intentionally changed-output manifest.
- Archive path, bytes, archive SHA-256, member SHA-256s, section offsets, and
  section SHA-256s.
- Runtime-consumption proof tying the archive to generated outputs.
- Strict compliance pass.
- Level-2 dispatch claim.
- Exact CUDA auth eval and adversarial review.

Shortest wall-clock path:

1. Debug the first HPM1 entropy mismatch, then rerun full decode/reencode.
2. Prove the runtime consumes the decoded categorical stream.
3. Run hidden-gem readiness and strict compliance on the exact packet.
4. Claim and dispatch only after parity/runtime proof is green.

Dispatch decision:

- Do not dispatch the current candidate. It lacks parity and runtime proof.

### 5. Gate `wr01_apply_pr106x_half` through component-response proof before any exact spend

Type: scorer-changing/output-changing packet. Not rate-only.

Why it ranks fifth:

- Hidden-gem readiness records a concrete archive candidate:
  `bytes=186222`, archive SHA
  `d2208fd18cbd2faef454f2bc46617af546b4e2ec9d509e82e5357577a19f2953`.
- It changes decoded output, so it is not judged by rate-only floor logic.
- Because it is above the active byte floor and has only a small byte-side
  effect, it needs component improvement to pay for the rate penalty.

Exact blockers:

- Strict preflight not ready.
- Component/adversarial evidence missing.
- Exact CUDA missing.
- No proof yet that the output change improves SegNet/PoseNet enough to offset
  the byte penalty over `185578`.
- No Level-2 claim.

Required proof/artifacts:

- Runtime-consumption proof and output-change manifest.
- Component-response smoke that is not promoted as score evidence but justifies
  exact spend.
- Strict pre-submission compliance pass.
- Level-2 claim.
- Exact CUDA auth eval and adversarial review.

Shortest wall-clock path:

1. Run the smallest component-response/provenance gate that proves the changed
   output is intentional and not a no-op.
2. Compare required non-rate improvement against the byte penalty.
3. If component-response is positive and compliance is clean, claim and run a
   single exact CUDA eval.

Dispatch decision:

- Do not dispatch as-is. It is above the byte floor and lacks component proof.

## Explicit Do-Not-Dispatch Set

These artifacts are useful for custody, conformance, parser proof, or future
stacking, but should not consume exact-CUDA dispatch now.

| Candidate | Class | Bytes | Dispatch refusal |
| --- | --- | ---: | --- |
| `pr106x_lgblock16_monolithic_section_candidate` | rate-only | `186079` | Runtime proof is now present, but it is `501` bytes above the `185578` A++ floor and has no active claim. Do not dispatch unless stacked onto the active floor or shrunk below it. |
| `hdm3` | rate-only | `186066` | `488` bytes above the active floor. Do not dispatch as a score-lowering candidate. |
| `pr106_q10_151byte_brotli` | rate-only | `186088` | `510` bytes above the active floor. Do not dispatch as a score-lowering candidate. |
| `pr101_schema_recode` / split-brotli family | rate-only or runtime-adapter probe | `185998` to `186044` | Above active floor and still missing runtime/proof coverage. Do not dispatch. |
| `pr101_codecop_lgwin18` | PR101-rate transducer | `178258` | Below byte floor but inherits PR101 exact semantic score near `0.226353`, worse than active `0.208981`. Manifest says `ready_for_exact_eval_dispatch=false` due missing runtime parity and claim. Use for conformance only unless stacked with a scorer-changing packet. |
| `omega_opt_linear_stack_packet_20260508` | prediction/planning | none | No archive bytes/path/SHA, no runtime packet, no exact anchor, and no layer runtime consumption. Do not dispatch predictions. |
| `cross_paradigm_admm_continuous_k_then_op1_finalizer_substrate_corrected` current row | byte proxy | `137469` | High-upside build target, but current row has no real scored archive/runtime proof. Do not dispatch until rebuilt as a contest packet. |
| `lossy_coarsening_analytical` measured config | exact negative | `156404` | A-negative at `score=0.351718793322788`; measured config retired. Do not redispatch without reactivation evidence. |
| `joint_admm_balle_arithmetic_stack` current state | scorer-changing stack | n/a | Runtime detects/refuses the `jcsp.bin` path and side information is not proven charged/consumed. Build runtime consumer first. |

## Watchlist After Top 5

- `joint_admm_balle_arithmetic_stack`: good architecture-level target, but it
  is behind runtime-consumer work. Promote it after `submissions/robust_current`
  actually consumes `jcsp.bin` and all side information is charged.
- `hnerv_per_tensor_context_entropy`: useful byte/grammar work, but raw-equal
  fixtures were byte-negative versus source Brotli and need a decoder runtime
  before exact-eval relevance.
- Geometry/foveation/RAFT/LA-POSE rows in the cross-paradigm inventory remain
  planning evidence until they emit charged bytes consumed by the scored
  runtime.

## Read-Only Audit Commands Used

- `git status --short --branch`
- `rg` over `AGENTS.md`, `CLAUDE.md`, `.omx/research`, `reports`, and
  `experiments/results` for the named targets.
- `sed`/`nl` reads of the relevant ledgers and manifests.
- `jq` reads of exact-eval JSON, readiness JSON, cathedral catalog/evidence,
  active Lightning state, and candidate manifests.

No pytest or code-changing command was run.

## Bottom Line

The next exact score-lowering attempt should be the active arch_shrink harvest,
not another launch. The next build target should be corrected cross-paradigm
ADMM-K plus Op1 as a real runtime-consumed contest packet. The tempting
rate-only monolithic Brotli artifacts are explicitly not dispatchable because
they sit above the `185578`-byte A++ floor, and PR101-only byte wins are not
score wins unless they overcome PR101's worse exact semantic terms.
