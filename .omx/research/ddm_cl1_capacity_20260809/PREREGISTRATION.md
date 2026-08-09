# DDM CL1 HPAC model-byte to token-byte ladder — preregistration

Status: **QUEUED-WITH-A-FIRE-ORDER; sandbox fire REFUSED**. This arm is
scorer-free and has not moved a score pointer. Axis for training is
`[macOS-MPS research-signal]`; all trainer byte figures are false-authority
until serialized. `score_claim=false`.

## Question and fixed reference

The tested question is whether allocating more serialized bits to PR130's HPAC
prior saves more serialized token bytes than it costs. The break-even condition
for adjacent rows is

`delta(real token bytes) / delta(real packed model bytes) < -1`.

The immutable reference is PR130 CPR1: 15,164 B standalone packed HPAC model,
114,852 B coder-table ideal tokens, and 116,980 B real Range tokens. Thus the two
reference joints are 130,016 B (packed model + ideal tokens) and 132,144 B
(packed model + real Range tokens). These are section-level rows, not full
archive sizes. PR130's contest row remains `S=0.172141297491896447`, 191,052 B,
archive SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.

## Why the first ladder is fixed-topology

The first ladder varies only the real trainer flag `--rate-lambda`. It holds
`channels=64`, `patch=64`, `delta=2`, and `frame_dim=8` fixed. Lower lambda lets
the learned per-output-channel bit depths spend more model bits to lower token
cross-entropy. That directly changes the x coordinate that matters—real packed
model bytes—without changing checkpoint tensor shapes, causal scan order, or the
receiver's hard-coded topology.

This ordering is load-bearing. An architectural width or frame-dimension rung
would need an owned initializer, a new float32 accumulator proof, and a matching
receiver configuration. Patch or delta additionally changes factorization and
token traversal. None is admitted before the fixed-topology byte-allocation
slope is measured.

## Preregistered ladder and fire order

All training uses the original 60-epoch cosine schedule, seed 20260716, batch 8,
QAT fraction 0.5, and the recovered byte-identical original DALI cache. No rung
uses the AV-decoded cache.

| Order | Rung | Lambda | Purpose | Fire trigger | Kill rule |
|---:|---|---:|---|---|---|
| 0 | `lambda_1p0_resume_control` | 1.0 | Literal SIGKILL after atomic epoch-1 publication, then exact resume through epoch 60 | ANS terminalized; lane claimed; P0 trainer committed | Any resume/config drift or missing complete checkpoint |
| 1 | `lambda_1p0_uninterrupted_twin` | 1.0 | Same-host training-repeat and interruption-equivalence control | Rung 0 packs and decodes exactly | Packed model or token output differs from rung 0; ladder remains blocked |
| 2 | `lambda_0p5` | 0.5 | First prior-growth rung, one log2 step below the reference lambda | Controls are byte-identical | Extra packed model bytes are not repaid by real Range token savings |
| 3 | `lambda_0p25` | 0.25 | Second prior-growth rung, another log2 step | Adjacent real Range slope from 1.0 to 0.5 is below -1 and decode is under 30 minutes | Adjacent slope is at least -1, receiver/decode fails, or joint bytes rise |

The values 1, 1/2, and 1/4 are a geometric Lagrange-multiplier bracket around
the shipped value, not measured optima. They are declared before looking at a
new rung. A higher-lambda shrink rung is not fired because the task is the
direction needed for the 33,252-byte gap: grow the prior only while net joint
bytes fall.

The estimand is fixed before fire: every lambda row uses the immutable
epoch-60 QAT stage checkpoint. This deliberately avoids a post-hoc epoch search
and is the smallest controlled lambda-to-serialized-bytes ladder. It does not
claim epoch 60 is the within-run byte optimum; an earlier checkpoint could have
a smaller real joint and remains outside this scoped first ladder.

The resume control is not a shortened algorithm. It runs the same 60-epoch
identity, is killed with SIGKILL only after the immutable epoch-1 checkpoint is
durable, and resumes with the live model, EMA shadow/update count, optimizer,
scheduler, best/history state, and every RNG restored. The uninterrupted twin
is a full independent repeat. Any causal or packed-byte difference blocks the
ladder; it is not silently folded into the slope.

## Frozen inputs and executable surfaces

| Object | Bytes | SHA-256 | Role |
|---|---:|---|---|
| Original DALI segmentation cache | 117,981,301 | `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195` | Every rung's exact n600 target field |
| P64 exact archive initialization | 165,367 | `0e6c30cef6b36c4e530779c92c56e9128c1d86c62e85e9fc5358a7e9f40ec985` | Every fixed-topology rung's common init |
| Shipped PR130 self-compress checkpoint | 177,041 | `0f4775920aeb2fb419555cc4d68703dd90b88be9d24c82466a99fddc1b1f1aa7` | Immutable external reference only |

The owned trainer is `tools/train_ddm_cl1_hpac_capacity.py`. It preserves the
intake model/loss/optimizer/scheduler equations and fail-closes on the pinned
source hashes. It adds the repository-mandated canonical EMA update/deployment
policy, atomic immutable periodic and continuous/QAT stage checkpoints,
`--resume-from`, a stable causal-state hash, embedded lineage plus ancestor
re-custody, full optimizer/scheduler/best/history/RNG custody, an SSD preflight,
and a success artifact manifest. The intake tree remains read-only. The EMA
adaptation means the lambda-1 controls establish the compliant local baseline;
the shipped PR130 bytes remain an immutable external section reference.

The same owned fitter is also the artifact runner. Its `pack`, `encode`, and
`decode` modes launch the exact pinned intake child without a shell, then write
an atomic attestation binding the runner, complete imported HPAC source closure,
runtime/environment, child argv, and all input/output/report bytes. Its `fit`
mode requires those attestations, their safe-run receipts, and the underlying
training receipts. The named resume control is inadmissible unless it proves a
real safe-run exit `-9`, an exact epoch-1 parent, fresh-root continuation,
embedded/preserved lineage, and a successful terminal trainer manifest.

The training result's `estimated_joint_bytes` is never a ladder row. It uses
direct-logit cross-entropy and theoretical variable-weight bits. Each selected
checkpoint must instead run through the pinned intake packer and the pinned
Range codec with their full structural argv.

## Row construction and fixed-horizon selection

For each completed training:

1. Preserve every two-epoch evaluation checkpoint and both stage boundaries as
   resumability/diagnostic evidence.
2. Exclude the trainer's surrogate-best file from ladder selection. It is not a
   full resume checkpoint and its theoretical model-bit objective is not the
   registered estimand.
3. Select only the immutable
   `qat_stage_end_epoch_0060.pt` full-state checkpoint.
4. Run that checkpoint through the owned attesting pack runner. The pinned
   child packer must report `verified_exact=true` and `max_logit_diff=0`.
5. Run the same checkpoint through the owned attesting n600 Range encode and
   exact decode modes. Record coder-table ideal bpp separately from the emitted
   token file's actual bytes; require the decoded raw token SHA and logit hashes
   to verify.

The resulting ladder columns are: lambda; selected epoch; exact packed model
bytes and SHA; coder-table ideal token bytes (derived with a stated ceiling from
`ideal_bpp * 600 * 384 * 512 / 8`); real Range token bytes and SHA; exact decode
boolean and token hash; decode wall time; ideal joint; real joint; delta from
PR130; and adjacent ideal/real slopes.

The final fitted slopes use ordinary least squares over the three unique lambda
representatives with x = exact packed model bytes and y = token bytes. The
duplicate lambda-1 control is never treated as a regression replicate. Ideal
and Range fits are separate. Their conditional descriptive 95% interval uses
the residual standard error and the df=1 Student-t critical value; the
lambda-1 exact-match control is reported separately. No OLS fit is emitted
until both controls and both non-reference rungs are real-packed and
real-encoded. If lambda 0.5 fails the exact Range secant, the family closes at
that reference boundary and lambda 0.25 is not fired.

The immutable PR130 15,164/114,852/116,980-B section row is excluded from the
controlled OLS but included in every model/token/joint delta and section-row
comparison. An observed lambda endpoint is never called a continuous knee.
When the first secant pays and the second does not, lambda 0.5 is only the best
observed terminal-QAT rung
inside a `[0.25, 1.0]` secant bracket; it is not a measured continuous optimum.
Two paying intervals yield `UNBRACKETED_LOWER_LAMBDA`.

Verdict scope is `FORMULATION: fixed-topology C64/P64/delta2/D8 learned
bit-allocation under the original 60-epoch recipe`. A slope at or above -1
closes further lower-lambda spending for this formulation. It does not, by
itself, kill architectural width or frame-conditioning capacity.

## Fire gates and current refusal

Fire requires all of the following at the same observation:

- the completed n600 ANS result/done receipt hashes still verify;
- no other active Metal work is found;
- lane `lane_ddm_cl1_hpac_capacity_20260809` has an active `local_metal` claim;
- the owned trainer and its tests are committed, with both Python review passes;
- `TAC_ADMISSION_ENFORCE=1`, `PYTHONHASHSEED=0`, and
  `PYTORCH_ENABLE_MPS_FALLBACK=0` are explicit on every heavy launch;
- the SSD preflight passes and every output stays below
  `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`;
- the first epoch runs through the governed launcher and produces a measured
  peak-RSS receipt; the continuation's cap is derived from that receipt; and
- a literal-SIGKILL MPS continuation from the epoch-1 checkpoint and its
  uninterrupted twin produce identical causal-state hashes and packed bytes;
  the continuation uses a fresh output root; and
- every terminal pack/encode/decode has a successful safe-run receipt plus an
  owned-runner attestation that still verifies every byte.

The raw detached ANS process is now terminal: result SHA-256
`8816f91afcc21060753a6612cda4e1b7f3b483a7aa073cbfa1b9b5d7e520d451`,
done-receipt SHA-256
`f099a42cb2990e06b0f4614b17f1ce737ce6e8a094ff02f41d7e5ffb4d97e5af`,
and former PID 89557 returns ESRCH. Its n600 encode-side result is ideal
114,851.8 B, Range 116,980 B, ANS 114,860 B, delta −2,120 B. This clears the
competing Metal gate. Independently, the Codex sandbox reports MPS built but
unavailable; the full canonical trainer invocation failed closed before output
creation instead of substituting CPU. MAIN/operator execution is required for
the real SIGKILL/resume positive control. Exact commands and custody steps are
in `MAIN_METAL_FIRE_ORDER.md`.

## RECALL EVIDENCE

The recall sweep searched content—not only filenames—across `.omx/research`,
`docs`, all `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, the
canonical task-status/arm queues, the canonical-equations registry, design/SPEC
files, and the full PR130 intake source/pipeline. Queries included:

```text
clean60|clean 60|60ep HPAC|60[- ]epoch HPAC
--channels|--patch|--delta|--frame-dim|--rate-lambda intersect HPAC|PR130
ddm_cl1|capacity ladder|model-token|model.*token.*ladder|PR130.*HPAC
hpac|pr130|clean60|model.*token|frame.*dim|rate.*lambda (equations registry)
```

No controlled same-object HPAC capacity ladder or empirical model-byte/token-byte
derivative was found in that bounded corpus. Beyond the charter seeds, recall
changed the plan in five ways:

- PR130's pipeline deliberately changed P32 to P64 but retained no comparable
  packed-model-plus-token row, so that lineage is a prior, not a rung.
- Codec defaults are P32/residual while the shipped recipe is P64/raw; every
  pack/encode/decode command must spell the full tuple.
- `rate_lambda` is a real nested model-bit allocation coordinate and needs no
  receiver change, so it is the safest first ladder.
- QAT joint estimates oscillated by 2,196 B and estimates have missed real
  packing by multiple kilobytes; therefore no terminal row is admitted until
  its actual model and Range stream are serialized. The fixed epoch-60 scope is
  a controlled estimand, not a claim of within-run optimality.
- The intake trainer's `.latest.pt` omits optimizer/scheduler/RNG/best state;
  its reset continuations were pathological, so the P0 lift is a pre-fire gate.

The canonical equations query returned only the same-object semantic-label
transfer equation, not a capacity derivative. FEED-07a's witness modulation
capacity warning concerns another vehicle and cannot kill this HPAC experiment.

## Current conclusion

No CL1 ladder row was measured in this sandbox, so `d(tokens)/d(model)` remains
absent and there is no knee verdict. The concrete result is a scoped,
machine-actionable fire order, a statically closed P0 trainer, and a fail-closed
fitter. P0 remains empirically unproved until the real MPS SIGKILL/resume pair.
The PR130 contest pointer and the live own-vehicle frontier are unchanged.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_cl1_capacity` MAIN Metal executor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_resume_control/`; fire trigger: zero competing Metal jobs, committed hashes/tests still verify, SSD/governor preflight PASS, and active claim for `lane_ddm_cl1_hpac_capacity_20260809`; execute the literal-SIGKILL control in `MAIN_METAL_FIRE_ORDER.md`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_cl1_capacity` harvester; consumer store: `.omx/research/ddm_cl1_capacity_20260809/` plus the SSD rung reports; fire trigger: resumed and uninterrupted lambda-1 controls have identical causal state, packed EMA bytes, Range bytes, and exact decodes; fire lambda 0.5, then conditionally lambda 0.25, and run the strict fitter.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_rc1_receiver`; consumer store: `src/tac/pr130_runtime/`; fire trigger: only if the fixed-topology lower-lambda result pays and a later arm proposes a structural rung; land config-aware receiver parse-back, accumulator, mask, and under-30-minute decode proof before that fire.

## LIVE-HYPOTHESES

- Lowering lambda from 1 to 0.5 may add fewer packed model bytes than it removes from the 116,980-byte Range token stream. This is plausible because the token section is 7.7 times the standalone packed model and the exchange has never been controlled.
- The best lambda may lie between the geometric rungs rather than at 0.5 or 0.25. This is plausible because QAT bit depths are discrete and the joint curve can kink when output channels cross integer bit-depth thresholds.
- Frame-conditioning capacity may pay even if fixed-topology bit allocation stalls. This is plausible because 600 distinct maps share the same small frame embedding, while increasing frame dimension adds less compute than widening the quadratic SPM pointwise layer.
- Replacing Range with the measured n600 ANS stream may recover 2,120 token bytes before any capacity change. This is plausible from the completed real-table encode measurement, but receiver integration and total archive bytes remain unproved.

## DEAD-ENDS

- Clean60 is closed as capacity evidence: it kept C64/P64/delta2/D8 fixed and lost by 185 B apples-to-apples.
- Trainer `bpp` or `estimated_joint_bytes` is closed as a packed row: neither serializes the actual model and Range token stream.
- Launching the intake trainer directly is closed: it cannot resume the optimizer trajectory and preserves no distinct phase checkpoint.
- Changing channels, patch, delta, or frame dimension before receiver adaptation is closed: IHS1 does not carry topology and the receiver hard-codes all four values.
- Codec defaults are closed: they silently choose P32 and residual targets, which is not the PR130 object.
- The ANS job as a current Metal blocker is closed: its wrapper returned rc=0, its result/done hashes verify, and former PID 89557 is ESRCH.
- An empty governor/claim summary by itself is closed as proof Metal is idle: the ANS launch demonstrated that raw detached work can be absent from both stores.
