# G57 — full-n600 direct two-layer public closure and exact failure boundary

Date: 2026-07-26  
Lane: `lane_original_taskspace_inverse_witness_codec_capstone_20260726`  
Axis: `[macOS-CPU advisory]`, frozen `upstream/evaluate.py`, batch 16  
Pointer: official upstream `0.172`, unchanged

## Outcome first

The first fresh, own-lineage, receiver-closed full-n600 two-stream control is
complete. Its exact public archive is `182,220 B`, SHA-256
`736d9c2e0313a51f79dddbdf2e899d69e8b8bcf5555c5801869464064cf27f98`.
Two distinct initially clean public inflates produced the same 1,200-frame,
`3,662,409,600 B` raw output, SHA-256
`ccddfd9ab3606ed7a1b8e6bc0d2213028e9408f0dc082c1cba62587118a2ca3d`.

The full upstream row is:

| component | value |
|---|---:|
| `d_seg` | `0.17946555` |
| `d_pose` | `45.10546494` |
| archive | `182,220 B` |
| exact rate term | `0.12133281843792207` |
| score from report-rounded components | `39.30593503092899` |
| upstream printed score | `39.31` |

This does not move the `0.172` frontier. No contest CPU/CUDA dispatch is
warranted for a row that is more than 39 score units above it.

## The critical claim correction

This is a real `DIRECT_TASK_LAYERED` control, not the literal G49/G50
`PROGRAM_RESIDUAL_LAYERED` experiment.

G51 materialized

`Yk[p] = round_u8(DisjointResizeOperator(gt_fk[p]))`

directly from current source frames. G52 then encoded temporal `Y1` and the
centered-q2 `Y0|Y1` enhancement at 23 kbps each. It did **not** consume a G49
selected-preimage packet or behavior-changing analytic/learned factors, and the
fresh 133,941-byte V15-derived semantic archive was not embedded, copied, or
charged. The production archive contains exactly the two fresh H.264 streams
and their manifest; historical payload contribution is zero.

That distinction is not bookkeeping. It explains the result. The direct plane
stream achieved a frontier-scale rate, but its quantization moved the output far
outside the frozen evaluator cells. The rate machinery worked; the represented
object was wrong for this byte budget.

## Why this should have been anticipated

The failure was visible before the final evaluator run, and advancing it as if
it were the requested selected-preimage arm was a pipeline error.

- G51's module-level contract and implementation explicitly say that it emits
  resized source RGB. Batch-16 labels were custody-bound but never used to
  choose the planes.
- G52 explicitly typed the result as `DIRECT_TASK_LAYERED` and refused
  `PROGRAM_RESIDUAL_LAYERED`; that should have been a hard semantic boundary,
  not a footnote.
- G52's final decoded-plane receipts already showed RMSE `8.7835` for `Y0` and
  `8.5172` for `Y1`, while each stream carried only about `0.00611` bits per
  scorer pixel. V10 preserves those altered planes exactly; it cannot project
  them back into an evaluator fiber.
- At 182,220 bytes, the rate leaves only `0.05066718` total distortion score.
  Even with impossible perfect Pose, `d_seg` had to be below `0.00050667`.
  The measured `0.17946555` is 354.2 times that best-case allowance.

The missing system invariant is therefore:

> rate-feasible bytes are not candidate admission; producer identity plus a
> full-n600 coupled scorer row on the exact decoded planes is candidate
> admission.

The first implementation encoded that invariant as a four-boundary linter:

1. `PRE_ENCODE` re-falsifies representation identity, counted semantic/program
   custody, behavior-changing factors, n600 coverage, and batch-16 geometry.
2. `PRE_PUBLIC_CLOSURE` computes the full coupled score from exact through-R
   components on the exact decoded raw object; rate-only feasibility cannot
   pass.
3. `PRE_PROMOTION` reopens receiver custody, two-clean-root determinism, full
   output shape, and contest CPU/CUDA authority.
4. `POST_EVAL` requires a formulation-scoped verdict and either real unified
   stack hooks or a typed integration blocker.

Replaying G57 through v1 produced `REFUSE`, `REFUSE`, `REFUSE`, then `ADMIT` for
the narrow post-eval learning record. That replay correctly *diagnosed* the
representation mismatch, but an independent adversarial review refused v1 as
an enforcement gate. The requests were authored after the eval, the frontier
and artifacts were self-attested, the four boundaries were not chained, the
requested representation changed after PRE_ENCODE refused, promotion could
admit the 39.31 row merely by changing its axis string, and no launcher enforced
the gate. Therefore it is false to say v1 anticipated or stopped G57.

V1 remains useful only as retrospective regression evidence. G59 v2 is the
owed production repair: live canonical-frontier reopening, canonical score
math, regular-file artifact reopening, one immutable campaign/representation/
archive/raw/eval chain, predecessor-admission enforcement, competitive score
before promotion, structured integration custody, and unavoidable governed
launcher hooks.

The exact G57 row is now registered in `.omx/state/probe_outcomes.jsonl` as
`g57_direct_task_layered_x264rgb_46k_n600_20260726`. Its `KILL` is explicitly
`FORMULATION`-scoped and advisory: it prevents redispatch of this equal-rate
direct-plane arm without killing selected-preimage factors, learned quotients,
unequal coupled allocation, or other representatives. The remaining canonical
learning hooks stay fail-closed rather than being represented by prose.

Future arms should build/stage once, run one full scorer pass, and perform the
second clean-root decode only after the score is competitive. More importantly,
the producer-identity gate must refuse to call direct source planes a
selected-preimage program.

## What is now settled

1. Public receiver closure is real. PyAV 17.0.0 parsed native `gbrp`; generic
   V10 factor-2 realization produced the expected video; both clean-root runs
   were deterministic and remained below the official 30-minute limit.
2. Container overhead is not the crux. The public archive has `180,264 B` of
   stream payload and only `1,956 B` of manifest plus ZIP overhead.
3. Equal 23 kbps direct pixel-plane coding is decisively dead. The distortion
   terms, not the `0.1213` rate term, create the `39.31` score.
4. This does **not** falsify selected-preimage coding, factorized semantic
   programs, learned quotients, or unequal task-costate allocation. None of
   those mechanisms was present in the measured archive.
5. A public archive/evaluator receipt can now be produced without re-solving
   receiver closure. The next experiment should reuse this generic closure and
   replace only the counted representation.

## Smallest missing production implementation

The missing edge is executable, not conceptual:

```text
fresh V15 semantic archive bytes (counted once)
  + G49 selected-preimage packet and behavior-changing factors (counted)
  -> reopen CarrierComposeReceiverV1
  -> iter_selected_preimage_segment, five segments of 120 pairs
  -> G52 operand-provider protocol
  -> residual/quotient Y1 and conditional Y0|Y1 streams
  -> existing G55 public archive and V10 factor-2 receiver
```

The bridge must refuse production use unless the outer archive charges the
fresh semantic bytes and factor packet, the factor set changes decoded pairs,
and any learned quotient binds a concrete generic decoder implementation. The
current G52 config correctly refuses to call itself `PROGRAM_RESIDUAL_LAYERED`.

After that bridge exists, the next rate decision must be coupled:

`S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489`.

Do not repeat equal layer bitrates. Measure the exact marginal score value of
base and enhancement bytes, and let the controller move bytes between them.

## Durable evidence

The machine-readable authority receipt is
`g57_direct_task_layered_fulln600_public_eval_receipt_20260726.json` in this
directory. It binds G51/G52/G55 source, config, archive, stream, parse-back,
double-decode, evaluator, report, and log hashes; it also records the exact
literal-task blocker and narrow verdict scope.

The four v1 regression receipts are
`g57_adversarial_{pre_encode,pre_public,pre_promotion,post_eval}_receipt_20260726.json`.
They are generated by `tools/audit_taskspace_codec_adversarial_gate.py` and are
body-hashed, strict-schema retrospective evidence. They are not live-admission
authority. Future codec arms must use the chained v2 reviews as enforced stage
boundaries rather than wait for an anomalous final score.

## Triality and unified-stack wire-in

- **DSL:** `DIRECT_TASK_LAYERED` and `PROGRAM_RESIDUAL_LAYERED` remain distinct
  types; a source-resize provider cannot silently satisfy the selected-preimage
  contract.
- **DAG:** G51 -> G52 -> G55 -> public double decode -> upstream batch-16 eval
  is closed. The unclosed edge is G49/fresh-semantic bytes -> G52 provider.
- **Equation:** the measured decomposition is
  `17.946555 + 21.238047212491075 + 0.12133281843792207 = 39.30593503092899`.
- **Sensitivity/costates:** direct base and enhancement need separate marginal
  score-per-byte observations; equal bitrate is not an admissible controller
  conclusion.
- **Pareto/bit allocator:** kill only this exact 46 kbps direct arm and feed its
  distortion/rate row into the whole-archive allocator.
- **Autopilot:** reuse the now-proven public receiver closure; dispatch next only
  when counted semantic and factor custody reopens end to end.

## Pointer honesty

Pointer delta is exactly zero. The official display remains `0.172`; this
advisory macOS CPU row is research-only and promotion-ineligible.
