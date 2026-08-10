# DDM TM1 — PR130 token-model lever findings

Date: 2026-08-09/10

Axis: `[macOS-CPU advisory, scorer-free]`

Score authority: `score_claim=false`

Measurement status: complete

Receiver status: blocked; the research archives are not evaluator-runnable

## Conclusion

The shipped `IntegerHPAC` cross-entropy baseline reproduced exactly enough to
clear the charter gate: **114,851.81025623773 B** over **117,964,800** real
tokens, displayed as **114,851.8 B**. The null-model implementation also
reproduced DT1's retained ANS stream byte-for-byte at **114,860 B**, SHA-256
`a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488`,
and decoded to the canonical raw-token SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.

Two decoder-computable priors crossed the charter's strict rate-mechanism
threshold after real ANS coding and real joint model-bundle compression:

- **Second-order temporal reversion:** 114,528 token B and +36 joint model B,
  a **296 B net reduction** against the exact ANS archive control. Its charter
  pair is 129,656 B, **287.810256 B below** the 129,943.810256-B target.
- **Confidence-conditioned residual:** 114,660 token B and +144 joint model B,
  a **56 B net reduction** against the exact ANS archive control. Its charter
  pair is 129,896 B, **47.810256 B below** target.

Both have positive seeded n120 five-fold out-of-fold evidence and exact n600
offline token reconstruction. The pre-registered falsifier therefore does not
fire. The token-model axis remains open.

This is a **rate mechanism**, not a submission row. The candidate ZIPs use a
research parser and retained causal logits; no owned `inflate.py` consumes the
new sidecar or ANS grammar yet. Consequently receiver closure, contest runtime,
held component identity through the literal receiver, and any score are not
measured. The exact pointer is unmoved.

## Candidate table

All token and model columns below are measured at n600. `Δmodel` is the exact
change in the jointly XZ-compressed full model bundle, not a raw or standalone
weight estimate. `Net` is the literal candidate-archive delta against the
188,932-B null-sidecar ANS research archive. OOF is the ideal-byte saving on one
seeded random frame from each of 120 five-frame strata; no held frame was used
to fit its fold model.

| Candidate | OOF saved B | Real token B | Δtoken vs ANS B | Δmodel B | Net B | Charter pair B | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| temporal reversion | 68.0594 | 114,528 | −332 | +36 | **−296** | **129,656** | supported rate mechanism |
| confidence LUT | 30.7188 | 114,660 | −200 | +144 | **−56** | **129,896** | supported rate mechanism |
| temperature | 0.2189 | 114,860 | 0 | +20 | +20 | 129,972 | exact configuration loses |
| class bias | ~0 | 114,860 | 0 | +24 | +24 | 129,976 | exact configuration loses |
| 30-frame block bias | ~0 | 114,860 | 0 | +40 | +40 | 129,992 | exact configuration loses |
| global 4×4 tile bias | 3.0677 | 114,848 | −12 | +84 | +72 | 130,024 | token gain loses after model bytes |

The six candidates represent five direction groups because temperature and
class bias share the global-calibration group. Each candidate selected its own
integer-lattice shrink/smoothing setting by five-fold OOF ideal bytes before the
final table was fit and real-coded at n600. These measurements do not close
other granularities or the wider architecture families.

## Winning mechanism

For each pixel in frame `t`, let `a` be its already decoded class at `t−1` and
`b` its already decoded class at `t−2`. If `a != b`, add the signed integer
correction `K[a,b]` to the current PR130 int16 logit code for class `b`; then use
the shipped float64-softmax-to-float32 probability-table contract. The selected
full-n600 correction table, in units of 1/8 logit, is:

```text
K =
[[ 0,  3,  1,  0,   3],
 [ 1,  0,  0, -5,   2],
 [ 2, 11,  0, -1,   0],
 [ 1,  6,  2,  0, -16],
 [ 3,  4,  0, -5,   0]]
```

The table is causal and decoder-computable. PR130 uses the same group-major
permutation in every frame, so the prior two decoded group-major vectors align
position-for-position with the current logits. The actual receiver may instead
retain row-major frames and gather through each current group mask. It must not
mix a row-major ravel directly with group-major logits.

The 20 off-diagonal int8 values plus format header occupy 29 raw sidecar bytes
and 39 packed bytes. When appended and recompressed with the complete model
bundle, the literal marginal is 36 B. The n120 OOF grid selected full Newton
strength (`shrink=1.0`). The n600 ideal token total is 114,518.189474 B and the
real ANS total is 114,528 B.

## Exact reconstruction and d_seg boundary

Every candidate ANS stream decoded all 117,964,800 symbols, emptied the ANS
terminal state, and reconstructed the canonical raw-token SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
The base semantic, carrier, and HPAC model bytes parsed back unchanged. Each
candidate archive was independently rebuilt and the repeat archive was
byte-identical.

These facts derive held `d_seg` **conditional on the measured research
parse-back contract**: unchanged renderer/model payload plus an identical token
tensor implies identical rendered semantics. They are not literal receiver
closure because the archive's sidecar/ANS adapter is not installed in the
owned runtime. No scorer was run, and the candidate archives report
`evaluator_runnable=false`.

## Runtime and storage

- Governed v2 measurement: 139.028 s, exit 0, no kill action.
- Null retained-table ANS encode/decode: 4.573 s / 4.161 s.
- Temporal retained-table encode/decode: 4.729 s / 4.349 s; decode delta
  **+0.188 s** against the same-host null control.
- Neural-forward growth in this formulation: none. The correction is an integer
  table lookup before probability construction.
- Reference full PR130 ANS receiver decode: 804.876493541 s
  `[macOS-CPU advisory, scorer-free]`, from DT1's retained receipt.
- Contest CPU/CUDA full-job runtime: not measured.
- The safe-run RSS receipt reports 0 MiB because process RSS inspection was
  unavailable in this restricted sandbox; it is not usable memory evidence.
- Durable v2 measurement store: 3.4 MiB on the SSD tier.

## Accounting conventions

The charter's strict pair uses the 15,092-B leave-one-out HPAC marginal and
therefore compares against 114,851.810256 + 15,092 = 129,943.810256 B. CL1's
standalone pack is 15,164 B and yields a different diagnostic baseline. The
candidate table uses the charter convention, while the exact literal full-ZIP
delta against the ANS-only archive is the rate-accounting authority.

Against the shipped 191,052-B Range archive, temporal reversion's research ZIP
is 188,636 B, a 2,416-B reduction. Of that, 2,120 B is the already measured ANS
replacement and only 296 B belongs to the new prior. The corresponding
all-else-held score projection is 0.1705325823 from PR130's 0.1721412975, but
this is false-authority projection only because the candidate receiver is not
implemented.

## RECALL EVIDENCE

Bounded searches included:

```text
rg -n -i 'IntegerHPAC|HPAC|PR130|token[-_ ]model|AR prior|autoregressive prior|model[-_ ]byte.*token|d\(tokens\)/d\(model\)|rate.lambda|frame.dim|patch_group_mask' .omx/research docs

.venv/bin/python tools/list_canonical_equations.py --json

rg -n -i 'ddm_(tm1|cl1|rr1|op1r)|IntegerHPAC|HPAC|PR130.*token|token.*PR130|model[-_ ]token|capacity ladder|rate.lambda' .omx/state/canonical_task_status.jsonl .omx/research/harness_tasklist_bridge_20260803.jsonl .omx/state/operator_p0_ledger.jsonl .omx/state/lane_registry.json

rg -c -i 'hpac|IntegerHPAC|token[-_ ]model|rate_lambda|frame_dim|clean60|ar[-_ ]prior' /Volumes/VertigoDataTier/pact/ddm_tm1_20260804/ddm_tm1_crossrun_frames_20260804.jsonl
```

What was found beyond the charter's seeds:

- `.omx/research/ddm_rr1_20260809/RECALL_AUDIT.md` had already completed a
  four-pass token-model audit.
- CL1 owns the fixed-topology `rate_lambda` ladder and has measured no rung;
  its apparatus landed at commit `fe8fa4f35e`.
- OP1R already owns the causal edge-context race. TM1 did not duplicate it.
- Clean60 is the same topology and loses by 185 B after corrected joint
  accounting; it is not a capacity direction.
- The older `ddm_tm1_20260804` store is cross-run telemetry mining, not a PR130
  token-model measurement.
- No qualifying same-object, real-coded, model-byte-counted prior variant was
  found in the bounded research/equation/task-store search.
- Current coordinates reset inside each P64 patch, leaving global position and
  richer temporal state genuinely unmeasured on this object.

This recall changed the plan: fixed-topology rate spending stayed with CL1,
edge context stayed with OP1R, and TM1 measured non-edge calibration, global
position, coarse frame state, and the newly derived t−2 causal context. It also
forced both the 15,092-B leave-one-out and 15,164-B standalone conventions to be
reported explicitly.

## Provenance and custody

Primary durable artifacts:

- Result:
  `/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/measurement_v2/tm1_result.json`,
  SHA-256 `8a0950601b309f4e1b18b5ec84001cd1d0901e90507aaa906e311dd429a00548`.
- Temporal archive:
  `/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/measurement_v2/candidates/temporal_reversion/archive.zip`,
  188,636 B, SHA-256
  `cc88b717f4abe3cc3874bc5dcf6ed55acdb8d983e1088ebfff76b761b08437c6`.
- Confidence archive:
  `/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/measurement_v2/candidates/confidence_lut/archive.zip`,
  188,876 B, SHA-256
  `4ccc9a77cd0289f68b43ec7bb0fdfdb9b0b0340877109d0972d8563e07832df4`.
- Run state SHA-256:
  `2d29a032c340d689e9074fab42b027075d7d718f0131f25a9e66c14f1c1b2781`.
- Governed launch manifest SHA-256:
  `c8cc5c9200586288f3080bf30481f13877d26a4d542d36027b8949194c39aac3`.
- Safe-run receipt SHA-256:
  `f031513aef32b82170980b8eeae0ec5d3a02b3f406e831ed49fd7ede37ea1e3f`.
- Resume-integrity audit: exit 0 in 1.061 s; result hash unchanged; safe-run
  receipt SHA-256
  `80f06e550e451bdfab1f25ea8738c5e2c0e0770454b308d8a2404afebae47abd`.
- Harness source SHA-256:
  `938b58620572639785b40b53f23319b448a1b570f2a736a52171e0675f15a906`.
- Focused-test source SHA-256:
  `26829d93a17bd2ad5ce6d843efd98f600cdfd4c9a0730503b161d22c2b195431`.

The input manifest was deep-hash checked across all 56 chunk files. The PR130
source definition came from the read-only intake at commit
`e34f31bc4969042c0051ac81aa3c56884419a231`; the public tag peels to
`2f94596bb0136d342254022a5c9584756eae0468`. DT1's retained-coder producer is
commit `0a0f402564d6ba45e3cc36835539d0e307bb036e`.

The first governed fire is preserved under
`/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/launch/` and
`measurement/`. It failed before a candidate row because two NumPy arrays used
an invalid `.square()` spelling. Both sites now use `np.square`, a regression
test covers the executed path, and v2 crossed that stage and completed. The
failed custody tree was not reused or deleted.

Tests at the measured harness hash: 17 passed; Ruff and bytecode compilation
passed.

## Pointer delta

No exact evaluator row was produced. The TM1 mechanism did not move any
frontier or pointer. Own-vehicle frontier remains S = 0.7539807296911207 at
357,836 B `[macOS-CPU advisory]`, n600.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `ddm_rc1_receiver` or a named TM1 receiver-closure successor; consumer store: `src/tac/pr130_runtime/` plus `/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/receiver_closure/`; fire trigger: `ddm_cx2` releases or explicitly accepts the mechanism and the receiver lane is claimed. Wire temporal reversion first, parse the exact archive twice, require canonical token SHA equality, measure bounded n600 receiver runtime, and only then expose it to a scorer consumer.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: future `ddm_tm1_combined` arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_tm1_20260809/combined_temporal_confidence/`; fire trigger: temporal receiver closure passes. Refit temporal plus confidence jointly with seeded n120 OOF selection, real joint model bytes, real ANS bytes, and exact receiver decode; do not add their standalone wins arithmetically.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN scorer scheduler; consumer store: a new exact-eval store named by MAIN; fire trigger: an evaluator-runnable temporal archive has two identical receiver parses, canonical token identity, bounded runtime, a free n600 scorer slot, and no active duplicate lane. Run `upstream/evaluate.py` on the exact archive bytes and keep contest CPU/CUDA axes separate.

## LIVE-HYPOTHESES

- Temporal reversion will retain most of its 296-B net win through the literal receiver because its only new state is the already decoded t−2 class and the offline decoder is exactly causal.
- The confidence LUT may compose with temporal reversion because it conditions on the base model's top class and margin rather than decoded temporal transitions; overlap is plausible, so only a joint real-code measurement can establish the sum.
- A 6×8 absolute-patch prior may beat the losing 4×4 tile instance because it aligns with PR130's actual P64 reset grid and can use a smaller gauge representation.
- Removal ablations for inherited frame/SPM components remain plausible because no matched same-object model-plus-token receipt prices their weight cost against token benefit.

## DEAD-ENDS

- Further entropy-coder tuning on the unchanged probability tables: ANS is already about eight bytes above ideal cross-entropy.
- Attributing the 2,120-B Range-to-ANS reduction to a new prior: the null ANS archive isolates it exactly.
- Global temperature, global class bias, 30-frame block bias, and the 4×4 tile table as measured here: none wins after its exact model bytes; this closes only those grid-selected configurations.
- Treating retained-logit parse-back as evaluator proof: no current `inflate.py` consumes the sidecar, so receiver/component/score claims remain blocked.
- Re-running CL1 `rate_lambda` or OP1R edge-context work under a TM1 name: those live directions already have owners.
- Reusing the failed v1 custody root: its source identity differs after the NumPy fix; v2 used a fresh root and the failure remains preserved.
